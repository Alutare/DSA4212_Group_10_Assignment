"""Evaluation: full-split loss / perplexity / bits-per-byte.

Validation is for tuning. Test is for FINAL numbers only and needs --final
(it refuses to overwrite an existing test_result.json unless --force).
"""
import argparse
import contextlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from gpt import GPT, GPTConfig
from utils import plan_windows, bits_per_byte, write_json_atomic


def make_ctx(dtype, device):
    if str(device).startswith("cuda") and dtype in ("float16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=getattr(torch, dtype))
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_stream(model, data, block_size, stride, batch_size, device, ctx, max_windows=None):
    """Returns (nll_sum_nats, n_scored_tokens). If max_windows is set, an evenly
    spaced FIXED subset of windows is used (same subset every call)."""
    T = block_size
    plan = plan_windows(len(data), T, stride)
    if max_windows and len(plan) > max_windows:
        idx = np.linspace(0, len(plan) - 1, max_windows).astype(int)
        plan = [plan[i] for i in idx]
    was_training = model.training
    model.eval()
    nll, n = 0.0, 0
    arange = torch.arange(T, device=device)[None, :]
    for i in range(0, len(plan), batch_size):
        chunk = plan[i:i + batch_size]
        x = np.stack([data[s:s + T] for s, _ in chunk]).astype(np.int64)
        y = np.stack([data[s + 1:s + 1 + T] for s, _ in chunk]).astype(np.int64)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        sf = torch.tensor([f for _, f in chunk], device=device)[:, None]
        with ctx:
            _, loss = model(x, y, reduction="none")
        mask = arange >= sf
        nll += (loss.float() * mask).sum().item()
        n += int(mask.sum().item())
    model.train(was_training)
    return nll, n


def metrics_from(nll, n, meta, split, stride=None):
    loss = nll / n
    return dict(loss=loss, ppl=math.exp(min(loss, 50)), bits_per_token=loss / math.log(2),
                bpb=bits_per_byte(nll, n, meta["tokens"][split], meta["bytes"][split]),
                n_scored=n, stride=stride)


def load_model_from_ckpt(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["model_args"]))
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    return model.to(device), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--stride", type=int, default=0, help="0 = block_size (non-overlapping)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.ckpt).parent
    if a.split == "test":
        if not a.final:
            raise SystemExit("Test split is reserved for final numbers: pass --final.")
        dest = out_dir / "test_result.json"
        if dest.exists() and not a.force:
            raise SystemExit(f"{dest} exists. Re-evaluating on test invites tuning on test; use --force if you must.")
    meta = json.loads((Path(a.data_dir) / "meta.json").read_text())
    data = np.memmap(Path(a.data_dir) / f"{a.split}.bin", dtype=np.uint16, mode="r")
    model, ck = load_model_from_ckpt(a.ckpt, a.device)
    T = model.config.block_size
    stride = a.stride or T
    nll, n = evaluate_stream(model, data, T, stride, a.batch_size, a.device, make_ctx(a.dtype, a.device))
    m = metrics_from(nll, n, meta, a.split, stride)
    m.update(split=a.split, ckpt=str(a.ckpt), tokenizer=meta["name"], block_size=T,
             params=model.param_counts(), train_state=ck.get("state"))
    print(json.dumps(m, indent=2, default=float))
    dest = out_dir / ("test_result.json" if a.split == "test" else f"eval_val_s{stride}.json")
    write_json_atomic(dest, m)
    print("wrote", dest)


if __name__ == "__main__":
    main()
