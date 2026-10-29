# Training Small Language Models under a Fixed Compute Budget
DSA4212 Group 10 — <names>

## 0. Declarations (required by the brief)
**Compute budget.** Each complete training run: **1800 s** of optimisation-step wall-clock on **one NVIDIA Tesla T4 (16 GB, Google Colab), fp16 mixed precision** (≈ <X> PFLOPs at measured MFU). Validation and checkpoint time are excluded; compile time is included. Resuming restores consumed time. Preliminary tuning runs (300 s) and tokenizer training are excluded. Budget declared on <date> before final runs.

**Initialisation.** Every model is randomly initialised (N(0, 0.02), GPT-2 residual scaling). No pretrained weights, embeddings or other pretrained neural components. Tokenizers: GPT-2 BPE merges (no neural component) and byte-level BPE trained by us on WikiText-103 train only. Implementation adapted from nanoGPT.

**Evaluation objective.** Held-out language modelling on WikiText-103 (raw): validation for all tuning and selection, test only for the final chosen configurations (each evaluated once). Metric: **bits-per-byte** (tokeniser-independent), plus loss and perplexity within a tokenizer. *Measures:* next-token predictive quality on encyclopedic English. *Leaves out:* downstream tasks, reasoning, other domains, long-range coherence beyond the context window, generation quality.

## 1. Introduction and research questions
1. Under a fixed budget, does a smaller model trained on more tokens beat a larger one trained on fewer? (Chinchilla, Hoffmann et al. 2022)
2. How does vocabulary size trade embedding cost against sequence length and speed?
3. Depth vs width at equal parameters. 4. (optional) context, batch size, precision/compile, repeated data (Muennighoff et al. 2023).

## 2. Setup
- Data: WikiText-103 raw, <N> train / <N> val / <N> test tokens (per tokenizer: table). Pre-processing convention.
- Model: pre-LN GPT, tied embeddings, learned positions; table of sizes with **total and non-embedding** parameters.
- Training: AdamW (β=0.9/0.95, wd 0.1), warm-up 100 steps then time-driven cosine to 10 %, grad-clip 1, 16 384 tokens/step, fp16 + GradScaler.
- Metrics and BPB formula. Hardware, seeds, code commit (`git_sha` in each result.json).

## 3. Throughput and budget allocation
Benchmark table (tokens/s, tokens in budget, epochs, tokens per parameter, MFU). Explain why a T4 gives low MFU.

## 4. Experiments and results
For every study: design (what varies, what is fixed), table (mean ± std over seeds where available), figure (`figures/<study>_summary.png`, `_curves.png`), 2–3 sentences of interpretation.
### 4.1 Tokenizer / vocabulary
### 4.2 Model size vs tokens  *(headline)*
Relate tokens-per-parameter at the optimum to the ≈20 tokens/param Chinchilla rule; discuss the epochs column (<1 epoch = not data-constrained).
### 4.3 Depth vs width
### 4.4 Efficiency (fp32 / fp16 / compile) and optimisation (LR, batch)
### 4.5 Context length / data repetition (if run)

## 5. Final test results
Table: chosen config(s), val BPB, **test BPB**, test perplexity, params, tokens seen. Seeds → mean ± std. A few generated samples (appendix).

## 6. Discussion
What mattered most per unit of compute? Surprises vs the reading list. Practical recipe for a 30-minute T4 budget.

## 7. Limitations
Colab speed variance; ≤3 seeds; confounds listed in README; single dataset/domain; LR tuned only at one size; BPB convention (appended newline); time budget ties results to this GPU/implementation.

## References
Hoffmann et al. 2022 (arXiv:2203.15556); Muennighoff et al. 2023 (arXiv:2305.16264); Karpathy nanoGPT; HuggingFace Smol Training Playbook; Merity et al. 2016 (WikiText).

## Appendix: reproducibility
`notebooks/DSA4212_Colab_Runner.ipynb`, commit <sha>, `figures/all_runs.csv`.
