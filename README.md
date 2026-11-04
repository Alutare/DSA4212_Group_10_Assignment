# DSA4212 Group 10 — Training small language models under a fixed compute budget

**Research question.** Under a fixed GPU-time budget, does a smaller Transformer trained on more tokens beat a larger one
trained on fewer? Secondary questions: vocabulary size / tokenizer, depth vs width, batch size and learning rate
(sensitivity of the headline result), repeated data, numerical precision.

**Declared budget (frozen before the final runs; see `code/experiments.py`).** 1800 s of optimisation-step wall-clock per run on one
NVIDIA Tesla T4 (16 GB, Google Colab), fp16 mixed precision. Validation / checkpoint time is excluded, `torch.compile` warm-up is
included, resuming restores the consumed time. 300 s tuning runs and tokenizer training are excluded.

**Initialisation.** Random only (N(0, 0.02), GPT-2 residual scaling). No pretrained weights or embeddings. Tokenizers: GPT-2 BPE merges via
`tiktoken` and byte-level BPE trained by us on WikiText-103 *train* only.

**Evaluation.** WikiText-103 (raw). Validation for tuning/selection, test only for the final runs (`evaluate.py --final`, one evaluation per run).
Metric: bits-per-byte (tokenizer independent) plus loss/perplexity within a tokenizer.

## Layout
```
readme.md  authors.md
code/      gpt.py train.py evaluate.py utils.py prepare_data.py train_tokenizer.py tok.py experiments.py
           run_experiments.py selection.py analyze.py benchmark.py baselines.py sample.py test_utils.py
notebooks/ DSA4212_Colab_Runner.ipynb   (end-to-end runner used for every number in the report)
results/   benchmark.csv, baselines.json, tuned_lr.json, all_runs.csv, summary_by_config.csv, final_test.csv, per-run result.json
report/    report source (LaTeX/Markdown), bibliography, figures
```

## Reproducing
1. Open `notebooks/DSA4212_Colab_Runner.ipynb` on a Colab T4 runtime. Put the four WikiText-103 raw parquet files
   (`train-0000{0,1}-of-00002`, `validation-00000-of-00001`, `test-00000-of-00001`; HF `wikitext`, `wikitext-103-raw-v1`) in `MyDrive/DSA4212/Dataset`.
2. Run the cells top to bottom. Every study is resumable and skips finished runs. Order: benchmark -> LR tuning (5 min runs) -> tokenizer study ->
   size study (headline) -> batch/LR sensitivity of the headline -> shape -> seeds -> data -> final test evaluation.
3. Local checks without a GPU: `cd code && python test_utils.py` and `python run_experiments.py --study size --dry_run`.
4. Environment: Colab default (Python 3.13, PyTorch 2.11 at the time of writing), plus `requirements.txt`. Versions and git SHA are stored in every `result.json`.

Raw checkpoints (`ckpt.pt`) are not committed (size). All `result.json` / csv summaries are in `results/`.
