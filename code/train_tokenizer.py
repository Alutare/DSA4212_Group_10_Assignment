"""Train a byte-level BPE tokenizer on WikiText-103 TRAIN text only.
(Tokenizer training is outside the per-run compute budget, as the brief allows.)
Byte-level + no normaliser => lossless, so bits-per-byte is exact.
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

TRAIN_FILES = ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with the parquet files")
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--out", required=True, help="e.g. tokenizers/bpe8k.json")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all training rows")
    a = ap.parse_args()

    def texts():
        seen = 0
        for fn in TRAIN_FILES:
            pf = pq.ParquetFile(Path(a.data_dir) / fn)
            for batch in pf.iter_batches(batch_size=4096, columns=["text"]):
                for t in batch.column(0).to_pylist():
                    if t:
                        yield t
                        seen += 1
                        if a.max_rows and seen >= a.max_rows:
                            return

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab_size, special_tokens=[],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=True)
    tok.train_from_iterator(texts(), trainer=trainer)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(a.out)
    s = "The quick brown fox jumped over 13 lazy dogs.\n = Valkyria Chronicles III = \n"
    ids = tok.encode(s).ids
    assert tok.decode(ids) == s, "tokenizer is not lossless!"
    print(f"saved {a.out}: vocab={tok.get_vocab_size()}, sample -> {len(ids)} tokens, lossless OK")


if __name__ == "__main__":
    main()
