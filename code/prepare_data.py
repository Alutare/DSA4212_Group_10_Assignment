"""Tokenise WikiText-103 (raw parquet) into uint16 .bin files + meta.json.

Convention (same for every tokenizer): each non-null row is encoded, then the
encoding of "\n" is appended. Bytes per split are counted on EXACTLY that
stream (utf-8 bytes of each row + 1 per appended newline), so bits-per-byte
= total bits / those bytes is comparable across tokenizers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tok import Tok

FILES = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "val": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with the parquet files")
    ap.add_argument("--tokenizer", default="gpt2", help="'gpt2' or path to a tokenizer json")
    ap.add_argument("--name", required=True, help="output sub-folder, e.g. gpt2 or bpe8k")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--batch_rows", type=int, default=2048)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists() and not a.force:
        print("already prepared:", out)
        return

    tok = Tok(a.tokenizer)
    assert tok.n_vocab <= 65535, "uint16 storage needs vocab <= 65535"
    nl = tok.encode("\n")
    assert nl, "newline must tokenise to at least one token"
    nl_bytes = len("\n".encode("utf-8"))
    meta = dict(name=a.name, tokenizer=a.tokenizer, n_vocab=tok.n_vocab, dtype="uint16",
                newline_ids=nl, tokens={}, bytes={}, rows={}, roundtrip_mismatches={},
                convention="row text tokenised, then '\\n' tokens appended; bytes = utf8(row)+1")

    for split, files in FILES.items():
        dest, tmp = out / f"{split}.bin", out / f"{split}.bin.partial"
        n_tok = n_bytes = rows = mism = checked = 0
        with open(tmp, "wb") as f:
            for fn in files:
                pf = pq.ParquetFile(Path(a.data_dir) / fn)
                for batch in pf.iter_batches(batch_size=a.batch_rows, columns=["text"]):
                    texts = [t for t in batch.column(0).to_pylist() if t is not None]
                    encs = tok.encode_batch(texts)
                    if checked < 5000:                      # losslessness spot-check
                        for t, e in list(zip(texts, encs))[:500]:
                            mism += int(tok.decode(e) != t)
                            checked += 1
                    ids = []
                    for t, e in zip(texts, encs):
                        ids.extend(e)
                        ids.extend(nl)
                        n_bytes += len(t.encode("utf-8")) + nl_bytes
                    arr = np.asarray(ids, dtype=np.uint16)
                    assert int(arr.max()) < tok.n_vocab
                    arr.tofile(f)
                    n_tok += len(arr)
                    rows += len(texts)
        tmp.replace(dest)
        meta["tokens"][split], meta["bytes"][split] = n_tok, n_bytes
        meta["rows"][split], meta["roundtrip_mismatches"][split] = rows, f"{mism}/{checked}"
        print(f"{a.name}/{split}: {rows:,} rows, {n_tok:,} tokens, {n_bytes:,} bytes, "
              f"{n_bytes / n_tok:.3f} bytes/token, roundtrip mismatches {mism}/{checked}")

    meta["bytes_per_token"] = {s: meta["bytes"][s] / meta["tokens"][s] for s in FILES}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out / "meta.json")


if __name__ == "__main__":
    main()
