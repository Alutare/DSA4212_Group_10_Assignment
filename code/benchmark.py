"""Throughput benchmark (run BEFORE the studies): tokens/s, memory, and how many
tokens / epochs / tokens-per-parameter each model gets in the declared budget.
Uses synthetic token batches, so it is a slight over-estimate of real throughput."""
import argparse
import csv
import time

import torch

import experiments as E
from gpt import GPT, GPTConfig


def bench(spec, vocab, steps, warmup, dtype, compile_):
    dev = "cuda"
    cfg = GPTConfig(block_size=spec["block_size"], vocab_size=vocab, n_layer=spec["n_layer"],
                    n_head=spec["n_head"], n_embd=spec["n_embd"])
    model = GPT(cfg).to(dev)
    opt = model.configure_optimizers(0.1, 1e-4, (0.9, 0.95), "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
    m = torch.compile(model) if compile_ else model
    B, T = spec["batch_size"], spec["block_size"]
    x = torch.randint(0, vocab, (B, T), device=dev)
    y = torch.randint(0, vocab, (B, T), device=dev)
    torch.cuda.reset_peak_memory_stats()

    def step():
        opt.zero_grad(set_to_none=True)
        for _ in range(spec["grad_accum"]):
            with torch.autocast("cuda", dtype=getattr(torch, dtype), enabled=(dtype != "float32")):
                _, loss = m(x, y)
            scaler.scale(loss / spec["grad_accum"]).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tps = steps * B * T * spec["grad_accum"] / dt
    pc = model.param_counts()
    del model, opt, m
    torch.cuda.empty_cache()
    return pc, tps, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=list(E.SIZES))
    ap.add_argument("--vocabs", nargs="+", type=int, default=[50257, 16384])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--train_tokens", type=float, default=119.7e6, help="size of the training stream")
    ap.add_argument("--out", default="benchmark.csv")
    a = ap.parse_args()
    print(f"budget = {E.BUDGET_SECONDS}s | GPU = {torch.cuda.get_device_name(0)}")
    rows = []
    hdr = f"{'size':>5} {'vocab':>6} {'total M':>8} {'non-emb M':>10} {'tok/s':>9} {'mem GB':>7} " \
          f"{'tokens in budget (M)':>21} {'epochs':>7} {'tok/param(ne)':>14}"
    print(hdr)
    for s in a.sizes:
        for v in a.vocabs:
            try:
                pc, tps, mem = bench(E.cfg(s), v, a.steps, a.warmup, a.dtype, a.compile)
            except torch.cuda.OutOfMemoryError:
                print(f"{s:>5} {v:>6}  OOM -> lower batch_size / raise grad_accum in experiments.py")
                torch.cuda.empty_cache()
                continue
            toks = tps * E.BUDGET_SECONDS
            row = dict(size=s, vocab=v, params_total=pc["total"], params_nonemb=pc["non_embedding"],
                       tok_per_s=tps, mem_gb=mem, tokens_in_budget=toks, epochs=toks / a.train_tokens,
                       tokens_per_param_nonemb=toks / pc["non_embedding"])
            rows.append(row)
            print(f"{s:>5} {v:>6} {pc['total'] / 1e6:8.2f} {pc['non_embedding'] / 1e6:10.2f} {tps:9,.0f} "
                  f"{mem:7.2f} {toks / 1e6:21.1f} {row['epochs']:7.2f} {row['tokens_per_param_nonemb']:14.1f}")
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("saved", a.out)


if __name__ == "__main__":
    main()
