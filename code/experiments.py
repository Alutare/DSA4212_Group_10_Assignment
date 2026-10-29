"""Single source of truth for the declared budget and all experiment grids.

>>> FREEZE BUDGET_SECONDS BEFORE RUNNING FINAL EXPERIMENTS AND DECLARE IT IN THE REPORT. <<<
"""
import math

# ---------------- declared compute budget (per complete training run) -------------
BUDGET_SECONDS = 1800          # 30 min of optimisation-step wall-clock per run ...
HARDWARE = "1x NVIDIA Tesla T4 (16 GB), Google Colab, fp16 mixed precision"   # ... on this
TUNE_FRACTION = 1 / 6          # preliminary tuning runs use 5 min (excluded from budget)

TOKENIZERS = {"gpt2": 50257, "bpe4k": 4096, "bpe8k": 8192, "bpe16k": 16384, "bpe32k": 32768}

# tokens/step is held at 16,384 for every size (micro-batch x accum x block)
BASE = dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=32, grad_accum=2,
            lr=8e-4, dropout=0.0, dtype="float16")

SIZES = {   # non-embedding params ~ 12 * L * E^2
    "xs": dict(n_layer=4, n_head=4, n_embd=128, lr=1.5e-3),     # ~0.8M
    "s":  dict(n_layer=4, n_head=4, n_embd=256, lr=1.0e-3),     # ~3.1M
    "m":  dict(n_layer=6, n_head=6, n_embd=384, lr=8.0e-4),     # ~10.6M
    "l":  dict(n_layer=8, n_head=8, n_embd=512, lr=6.0e-4),     # ~25M
    "xl": dict(n_layer=12, n_head=12, n_embd=768, lr=4.0e-4,    # ~85M
               batch_size=16, grad_accum=4),
}

SHAPES = {  # ~10M non-embedding params, different depth/width trade-off
    "wide3":   dict(n_layer=3, n_head=9, n_embd=576),
    "base6":   dict(n_layer=6, n_head=6, n_embd=384),
    "deep12":  dict(n_layer=12, n_head=4, n_embd=256),
    "deep24":  dict(n_layer=24, n_head=3, n_embd=192),
}


def cfg(size="m", **over):
    d = dict(BASE)
    d.update(SIZES[size])
    d.update(over)
    return d


def build_runs(study, tok="gpt2", size="m"):
    """Return list of dict(config=<name>, tok=<tokenizer folder>, args={...})."""
    R = []
    if study == "tokenizer":
        for t in TOKENIZERS:
            R.append(dict(config=f"tok_{t}", tok=t, args=cfg("m")))
    elif study == "size":
        for s in SIZES:
            R.append(dict(config=f"size_{s}_{tok}", tok=tok, args=cfg(s)))
    elif study == "shape":
        for name, sh in SHAPES.items():
            R.append(dict(config=f"shape_{name}_{tok}", tok=tok, args=cfg("m", **sh)))
    elif study == "context":
        for T, (bs, ga) in {128: (64, 2), 256: (32, 2), 512: (16, 2), 1024: (8, 2)}.items():
            R.append(dict(config=f"ctx_{T}_{tok}", tok=tok, args=cfg(size, block_size=T, batch_size=bs, grad_accum=ga)))
    elif study == "lr":
        base = cfg(size)
        for lr in [2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3]:
            R.append(dict(config=f"lr_{lr:.1e}_{size}_{tok}", tok=tok, args=dict(base, lr=lr)))
    elif study == "batch":      # lr scaled by sqrt(batch ratio); stated as an assumption in the report
        base = cfg(size)
        for tps, (bs, ga) in {4096: (16, 1), 16384: (32, 2), 65536: (32, 8)}.items():
            R.append(dict(config=f"batch_{tps}_{size}_{tok}", tok=tok,
                          args=dict(base, batch_size=bs, grad_accum=ga, lr=base["lr"] * math.sqrt(tps / 16384))))
    elif study == "efficiency":
        R.append(dict(config=f"eff_fp32_{size}_{tok}", tok=tok, args=cfg(size, dtype="float32")))
        R.append(dict(config=f"eff_fp16_{size}_{tok}", tok=tok, args=cfg(size)))
        R.append(dict(config=f"eff_fp16compile_{size}_{tok}", tok=tok, args=cfg(size, compile=True)))
    elif study == "data":       # repeated data: small model sees >1 epoch, shrink the corpus
        for frac in [0.01, 0.03, 0.1, 0.3, 1.0]:
            R.append(dict(config=f"data_{frac}_xs_{tok}", tok=tok, args=cfg("xs", train_fraction=frac)))
    elif study == "seeds":      # repeat your best config; pass --size and --tok
        R.append(dict(config=f"best_{size}_{tok}", tok=tok, args=cfg(size)))
    else:
        raise ValueError(f"unknown study {study}")
    return R


STUDIES = ["tokenizer", "size", "shape", "context", "lr", "batch", "efficiency", "data", "seeds"]
