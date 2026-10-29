"""Generate text from a trained checkpoint (qualitative samples for the report)."""
import argparse
import json
from pathlib import Path

import torch

from evaluate import load_model_from_ckpt, make_ctx
from tok import Tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tokenizer", default=None, help="override tokenizer spec stored in meta.json")
    ap.add_argument("--prompt", default=" = The history of Singapore = \n")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    meta = json.loads((Path(a.data_dir) / "meta.json").read_text())
    tok = Tok(a.tokenizer or meta["tokenizer"])
    model, _ = load_model_from_ckpt(a.ckpt, a.device)
    model.eval()
    T = model.config.block_size
    idx = torch.tensor([tok.encode(a.prompt)], device=a.device)
    ctx = make_ctx(a.dtype, a.device)
    with torch.no_grad():
        for _ in range(a.tokens):
            with ctx:
                logits, _ = model(idx[:, -T:])
            logits = logits[:, -1, :].float() / max(a.temperature, 1e-6)
            if a.top_k:
                v, _ = torch.topk(logits, min(a.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(torch.softmax(logits, -1), 1)], dim=1)
    print(tok.decode(idx[0].tolist()))


if __name__ == "__main__":
    main()
