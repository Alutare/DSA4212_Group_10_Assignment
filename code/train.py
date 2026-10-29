"""Time-budgeted training of a GPT from RANDOM initialisation.

* The budget is wall-clock TRAINING time (--budget_seconds). Only the time spent
  inside optimisation steps (data fetch + fwd + bwd + optimiser, GPU-synchronised)
  is counted. Periodic validation and checkpoint writing are NOT counted.
  torch.compile warm-up and CUDA start-up ARE counted (they are part of the run).
* The cosine LR schedule is driven by the fraction of the time budget used, so
  every run decays to min_lr exactly when the budget ends.
* Resuming from ckpt.pt restores the consumed budget (train_time), so resuming
  never resets the allowance.
"""
import argparse
import contextlib
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from evaluate import evaluate_stream, metrics_from, make_ctx
from gpt import GPT, GPTConfig
from utils import lr_at, read_jsonl, write_json_atomic, git_sha


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--config_name", default=None, help="run name without seed (for aggregation)")
    p.add_argument("--study", default="adhoc")
    p.add_argument("--stage", default="final", help="free tag: tune | final | smoke")
    p.add_argument("--data_dir", required=True, help="folder made by prepare_data.py")
    p.add_argument("--out_dir", required=True)
    # model
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    # optimisation
    p.add_argument("--batch_size", type=int, default=32, help="micro-batch (sequences)")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # budget
    p.add_argument("--budget_seconds", type=float, default=1800.0)
    p.add_argument("--max_steps", type=int, default=0, help="0 = off (debug only)")
    # data
    p.add_argument("--train_fraction", type=float, default=1.0,
                   help="use only the first fraction of the training stream (data study)")
    # system
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--peak_tflops", type=float, default=65.0, help="T4 fp16 tensor peak, for MFU")
    p.add_argument("--seed", type=int, default=1337)
    # logging / eval
    p.add_argument("--eval_every_seconds", type=float, default=120.0)
    p.add_argument("--eval_windows", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--ckpt_every_seconds", type=float, default=300.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--force", action="store_true", help="retrain even if result.json exists")
    return p.parse_args(argv)


def main():
    args = get_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() and not args.force:
        print(f"[{args.name}] already finished -> skipping")
        return

    dev = args.device
    cuda = dev.startswith("cuda")
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    T = args.block_size
    n_train = int(len(train_data) * args.train_fraction)
    assert n_train > T + 2, "train_fraction too small"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    cfg = GPTConfig(block_size=T, vocab_size=meta["n_vocab"], n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    raw_model = GPT(cfg).to(dev)                      # random init: no pretrained weights
    pc = raw_model.param_counts()
    flops_tok = raw_model.flops_per_token()
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.lr,
                                               (args.beta1, args.beta2), "cuda" if cuda else "cpu")
    scaler = torch.amp.GradScaler("cuda" if cuda else "cpu",
                                  enabled=(args.dtype == "float16" and cuda))
    ctx = make_ctx(args.dtype, dev)

    state = dict(step=0, train_time=0.0, tokens_seen=0)
    n_resumes = 0
    ck_path, log_path = out / "ckpt.pt", out / "log.jsonl"
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        assert ck["model_args"] == asdict(cfg), "checkpoint config differs from requested config"
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        n_resumes = ck.get("n_resumes", 0) + 1
        keep = [r for r in read_jsonl(log_path) if r.get("train_time", 0) <= state["train_time"]]
        log_path.write_text("".join(json.dumps(r) + "\n" for r in keep))
        print(f"[{args.name}] RESUMED at step {state['step']}, "
              f"{state['train_time']:.0f}s of {args.budget_seconds:.0f}s budget already used")
        del ck
    model = torch.compile(raw_model) if args.compile else raw_model

    tokens_per_step = args.batch_size * args.grad_accum * T
    print(f"[{args.name}] params total={pc['total']:,} non-embedding={pc['non_embedding']:,} "
          f"| tokens/step={tokens_per_step:,} | budget={args.budget_seconds:.0f}s | device={dev}")

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def get_batch(rng):
        ix = rng.integers(0, n_train - T, size=args.batch_size)
        x = torch.from_numpy(np.stack([train_data[i:i + T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([train_data[i + 1:i + 1 + T] for i in ix]).astype(np.int64))
        if cuda:
            return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)
        return x, y

    def save_ckpt(finished=False):
        ck = dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                  scaler=scaler.state_dict(), state=dict(state), model_args=asdict(cfg),
                  args=vars(args), finished=finished, n_resumes=n_resumes)
        tmp = out / "ckpt.pt.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, ck_path)

    def do_eval():
        nll, n = evaluate_stream(raw_model, val_data, T, T, args.eval_batch_size, dev, ctx,
                                 max_windows=args.eval_windows)
        m = metrics_from(nll, n, meta, "val", T)
        log(dict(type="val", step=state["step"], train_time=state["train_time"],
                 tokens=state["tokens_seen"], loss=m["loss"], bpb=m["bpb"]))
        print(f"  [val] step {state['step']:6d} | {state['train_time']:7.1f}s | "
              f"loss {m['loss']:.4f} | bpb {m['bpb']:.4f}")

    last_eval = state["train_time"] if n_resumes else -1e9
    last_ckpt = state["train_time"]
    diverged = False
    win_t, win_tok = 0.0, 0
    raw_model.train()
    while True:
        if state["train_time"] >= args.budget_seconds:
            break
        if args.max_steps and state["step"] >= args.max_steps:
            break
        if state["train_time"] - last_eval >= args.eval_every_seconds:
            do_eval()
            last_eval = state["train_time"]
        if state["train_time"] - last_ckpt >= args.ckpt_every_seconds:
            save_ckpt()
            last_ckpt = state["train_time"]

        # ----------------------- timed region -----------------------
        t0 = time.perf_counter()
        lr = lr_at(state["step"], state["train_time"] / args.budget_seconds, args.lr,
                   args.lr * args.min_lr_ratio, args.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        rng = np.random.default_rng([args.seed, state["step"]])
        optimizer.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=dev)
        for _ in range(args.grad_accum):
            x, y = get_batch(rng)
            with ctx:
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            loss_acc += loss.detach()
            scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # ------------------------------------------------------------
        state["train_time"] += dt
        state["step"] += 1
        state["tokens_seen"] += tokens_per_step
        win_t += dt
        win_tok += tokens_per_step

        if state["step"] % args.log_every == 0 or state["step"] == 1:
            l = float(loss_acc)
            log(dict(type="train", step=state["step"], train_time=state["train_time"],
                     tokens=state["tokens_seen"], loss=l, lr=lr, tok_s=win_tok / max(win_t, 1e-9)))
            print(f"  step {state['step']:6d} | {state['train_time']:7.1f}s | loss {l:.4f} | "
                  f"lr {lr:.2e} | {win_tok / max(win_t, 1e-9):,.0f} tok/s")
            win_t, win_tok = 0.0, 0
            if not np.isfinite(l):
                print("!! non-finite loss: marking run as diverged")
                diverged = True
                break

    # ----------------------- finish -----------------------
    save_ckpt(finished=True)
    nll, n = evaluate_stream(raw_model, val_data, T, T, args.eval_batch_size, dev, ctx)  # full val
    val = metrics_from(nll, n, meta, "val", T)
    log(dict(type="final_val", step=state["step"], train_time=state["train_time"], **val))
    curve = [dict(step=r["step"], train_time=r["train_time"], tokens=r["tokens"], loss=r["loss"],
                  bpb=r["bpb"]) for r in read_jsonl(log_path) if r.get("type") == "val"]
    tt = max(state["train_time"], 1e-9)
    result = dict(
        name=args.name, config_name=args.config_name or args.name, study=args.study,
        stage=args.stage, seed=args.seed, args=vars(args),
        tokenizer_name=meta["name"], vocab_size=meta["n_vocab"],
        params=pc, flops_per_token=flops_tok,
        train_time=state["train_time"], steps=state["step"], tokens_seen=state["tokens_seen"],
        tokens_per_step=tokens_per_step, train_tokens_available=n_train,
        epochs=state["tokens_seen"] / n_train,
        tokens_per_param_total=state["tokens_seen"] / pc["total"],
        tokens_per_param_nonemb=state["tokens_seen"] / max(pc["non_embedding"], 1),
        tokens_per_sec=state["tokens_seen"] / tt,
        mfu=flops_tok * state["tokens_seen"] / (tt * args.peak_tflops * 1e12),
        peak_mem_gb=(torch.cuda.max_memory_allocated() / 1e9) if cuda else None,
        gpu=torch.cuda.get_device_name(0) if cuda else platform.processor(),
        torch=torch.__version__, git_sha=git_sha(Path(__file__).parent),
        diverged=diverged, n_resumes=n_resumes, val=val, val_curve=curve)
    write_json_atomic(out / "result.json", result)
    print(f"[{args.name}] DONE | val loss {val['loss']:.4f} | ppl {val['ppl']:.2f} | "
          f"bpb {val['bpb']:.4f} | {state['tokens_seen'] / 1e6:.1f}M tokens ({result['epochs']:.2f} epochs)")


if __name__ == "__main__":
    main()
