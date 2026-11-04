"""Non-neural reference points in bits-per-byte (BPB), computed on the SAME byte stream as the models
(each row's UTF-8 text followed by "\\n"; see prepare_data.py).

  * general-purpose compressors (gzip -9, bz2 -9, xz -9): bpb = 8 * compressed_bytes / raw_bytes.
    They adapt to the evaluated text itself, so they are a sanity anchor, not a held-out language model.
  * token unigram and interpolated bigram, fitted on the TRAIN split only, add-one smoothed unigram;
    the bigram interpolation weight is chosen on VAL and then reported on val and test.

  python baselines.py --parquet_dir PARQUET --data_root /content/data --tokenizers gpt2 bpe4k --out ../results/baselines.json
The bigram is only computed for vocabularies <= 8192 (dense V x V count matrix).
"""
import argparse
import bz2
import gzip
import json
import lzma
import math
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
FILES = {"val": ["validation-00000-of-00001.parquet"], "test": ["test-00000-of-00001.parquet"]}


def split_bytes(parquet_dir, split):
    import pyarrow.parquet as pq
    parts = []
    for fn in FILES[split]:
        pf = pq.ParquetFile(Path(parquet_dir) / fn)
        for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
            for t in batch.column(0).to_pylist():
                if t is not None:
                    parts.append(t.encode("utf-8") + b"\n")
    return b"".join(parts)


def compressor_bpb(raw):
    return {"gzip9": 8 * len(gzip.compress(raw, 9)) / len(raw),
            "bz2_9": 8 * len(bz2.compress(raw, 9)) / len(raw),
            "xz9": 8 * len(lzma.compress(raw, preset=9)) / len(raw)}


def count_unigram(train, V, chunk=10_000_000):
    c = np.zeros(V, dtype=np.int64)
    for i in range(0, len(train), chunk):
        c += np.bincount(np.asarray(train[i:i + chunk], dtype=np.int64), minlength=V)
    return c


def count_bigram(train, V, chunk=10_000_000):
    c = np.zeros(V * V, dtype=np.int64)
    for i in range(0, len(train) - 1, chunk):
        x = np.asarray(train[i:i + chunk + 1], dtype=np.int64)       # +1: pair spanning the chunk edge
        c += np.bincount(x[:-1] * V + x[1:], minlength=V * V)
    return c.reshape(V, V)


def unigram_nll(uni, data, V):
    p = (uni + 1.0) / (uni.sum() + V)
    x = np.asarray(data, dtype=np.int64)
    return float(-np.log(p[x[1:]]).sum()), len(x) - 1            # first token has no context: scored like the models


def bigram_nll(uni, big, data, V, lam):
    p_uni = (uni + 1.0) / (uni.sum() + V)
    x = np.asarray(data, dtype=np.int64)
    a, b = x[:-1], x[1:]
    ctx = uni[a].astype(np.float64)
    p_ml = np.divide(big[a, b], ctx, out=np.zeros(len(a)), where=ctx > 0)
    p = lam * p_ml + (1 - lam) * p_uni[b]
    return float(-np.log(p).sum()), len(a)


def to_bpb(nll, n, meta, split):
    return (nll / n) * (meta["tokens"][split] / meta["bytes"][split]) / LN2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--tokenizers", nargs="+", default=["gpt2", "bpe4k"])
    ap.add_argument("--out", default="baselines.json")
    ap.add_argument("--no_compress", action="store_true")
    a = ap.parse_args()
    res = {}
    if not a.no_compress:
        for split in ("val", "test"):
            res[f"compress_{split}"] = compressor_bpb(split_bytes(a.parquet_dir, split))
            print(f"compressors on {split}:", {k: round(v, 3) for k, v in res[f"compress_{split}"].items()})
    for name in a.tokenizers:
        d = Path(a.data_root) / name
        meta = json.loads((d / "meta.json").read_text())
        V = meta["n_vocab"]
        data = {s: np.memmap(d / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val", "test")}
        uni = count_unigram(data["train"], V)
        r = {"vocab": V}
        for s in ("val", "test"):
            r[f"unigram_{s}_bpb"] = to_bpb(*unigram_nll(uni, data[s], V), meta, s)
        if V <= 8192:
            big = count_bigram(data["train"], V)
            grid = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
            val_scores = {lam: to_bpb(*bigram_nll(uni, big, data["val"], V, lam), meta, "val") for lam in grid}
            lam = min(val_scores, key=val_scores.get)               # chosen on validation only
            r.update(bigram_lambda=lam, bigram_val_bpb=val_scores[lam],
                     bigram_test_bpb=to_bpb(*bigram_nll(uni, big, data["test"], V, lam), meta, "test"))
        res[name] = r
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
