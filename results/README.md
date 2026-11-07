# Experimental Results - DSA4212 Group 10

This directory contains the training results, evaluation metrics, experiment logs, benchmark outputs, and figures for our project, *Training Small Language Models under a Fixed Compute Budget*.

All results were generated using WikiText-103 and GPT-style Transformer models trained from scratch on an NVIDIA Tesla T4 GPU.

## 1. Experiment Summary

We completed **36 training runs** across five experimental studies.

| Study | Description | Runs | Stage |
|---|---|---:|---|
| A | Learning-rate tuning | 17 | `tune/lr` |
| B | Tokenizer comparison | 5 | `final/tokenizer` |
| C | Model-size comparison | 5 | `final/size` |
| D | Batch-size sensitivity | 3 | `final/batch` |
| E | Random-seed robustness | 6 | `final/seeds` |
| **Total** | | **36** | |

The 17 learning-rate runs include two additional experiments at LR = 0.0064 for XS and M.

The seed robustness study contains three runs each for M and L, using seeds 1337, 1338, and 1339.

## 2. Final Test Evaluation

We completed **16 held-out test evaluations**, using saved checkpoints from the final training experiments.

| Study | Test Evaluations |
|---|---:|
| Tokenizer | 5 |
| Model size | 5 |
| Seed robustness | 6 |
| **Total** | **16** |

Batch-size sensitivity experiments were not included in the final test evaluation.

Test evaluations were conducted without retraining the models.

## 3. Key Results

### Model-Size Comparison

Each model was trained for 1,800 seconds using BPE 16K.

| Model | Non-Embedding Parameters | Tokens Seen | Validation BPB ↓ | Test BPB ↓ |
|---|---:|---:|---:|---:|
| XS | 0.79M | 491.36M | 1.29073 | 1.31015 |
| S | 3.15M | 346.57M | 1.18622 | 1.20683 |
| **M** | **10.62M** | **187.43M** | **1.12613** | **1.14615** |
| L | 25.17M | 97.17M | 1.13750 | 1.15642 |
| XL | 84.95M | 37.26M | 1.25302 | 1.27097 |

M achieved the lowest validation and test BPB among the five sizes.

### Tokenizer Comparison

| Tokenizer | Validation BPB ↓ | Test BPB ↓ |
|---|---:|---:|
| GPT-2 | 1.17927 | 1.20380 |
| BPE 4K | 1.17605 | 1.19205 |
| BPE 8K | 1.14428 | 1.16269 |
| **BPE 16K** | **1.12217** | **1.14468** |
| BPE 32K | 1.12896 | 1.15296 |

BPE 16K was selected based on validation BPB and retained the lowest BPB in the held-out tokenizer test comparison.

### Seed Robustness

| Model | Runs | Mean Validation BPB ↓ | Sample SD |
|---|---:|---:|---:|
| M | 3 | 1.13418 | 0.00181 |
| L | 3 | 1.14385 | 0.00062 |

M achieved a lower mean validation BPB, while L had lower observed variability.

The statistics are calculated separately for each model. Pooling all six seed runs into one standard deviation would conflate model-size differences with seed variability.

### Batch-Size Sensitivity

The XL model was tested with three effective batch sizes.

| Tokens per Step | Learning Rate | Validation BPB ↓ |
|---|---:|---:|
| 4,096 | 0.0008 | 1.31192 |
| **16,384** | **0.0016** | **1.25055** |
| 65,536 | 0.0032 | 1.46325 |

Learning rates were scaled with the square root of the batch-size ratio. These results therefore represent joint batch-size and learning-rate sensitivity rather than a batch-only ablation.

## 4. Directory Structure

```text
results/
├── README.md
├── all_experiments.csv
├── tuned_lr.json
├── headline_size_table.csv
├── tokenizer_stats.csv
├── seed_individual_results.csv
├── seed_robustness_summary.csv
├── final_test.csv
├── final_test_comparison.csv
├── benchmark.csv
├── benchmark_annotated.csv
├── benchmark_fp32.csv
├── benchmark_fp16_compile.csv
├── baselines.json
├── figures/
│   ├── all_runs.csv
│   ├── summary_by_config.csv
│   ├── tokenizer_summary.png
│   ├── tokenizer_curves.png
│   ├── size_summary.png
│   ├── size_curves.png
│   ├── batch_summary.png
│   ├── batch_curves.png
│   ├── seeds_summary.png
│   └── seeds_curves.png
└── runs/
    ├── tune/lr/
    └── final/
        ├── tokenizer/
        ├── size/
        ├── batch/
        └── seeds/
```

## 5. File Descriptions

### Consolidated Results

| File | Description |
|---|---|
| `all_experiments.csv` | Consolidated metadata and metrics for all 36 training runs |
| `headline_size_table.csv` | Main model-size comparison, including parameters, tokens, throughput, compute and validation BPB |
| `final_test.csv` | Consolidated held-out test evaluation results |
| `final_test_comparison.csv` | Validation versus test BPB for 16 evaluated configurations |
| `tuned_lr.json` | Learning rates used by the final training experiments |
| `tokenizer_stats.csv` | Vocabulary sizes, token counts and embedding-related statistics |
| `seed_individual_results.csv` | Individual M and L seed results |
| `seed_robustness_summary.csv` | Mean and standard deviation of validation BPB by model |

### Benchmarks and Baselines

- `benchmark.csv`: GPU benchmark measurements.
- `benchmark_annotated.csv`: Estimated throughput, token budgets and parameter efficiency.
- `benchmark_fp32.csv`: FP32 benchmark results.
- `benchmark_fp16_compile.csv`: FP16/compilation benchmark results.
- `baselines.json`: Compression and n-gram baseline measurements.

### Figures

The `figures/` directory contains training curves and summary plots for tokenizer, model-size, batch-size and seed studies.

`all_runs.csv` contains detailed final-run metrics, while `summary_by_config.csv` aggregates results by experiment configuration.

## 6. Individual Experiment Files

Each experiment has its own directory under `runs/`.

A completed training run typically contains:

```text
experiment_name/
├── result.json
├── log.jsonl
└── stdout.log
```

For runs included in the final test evaluation, an additional file is present:

```text
test_result.json
```

The files serve different purposes:

- `result.json`: Training configuration, validation metrics, throughput, parameter counts and training statistics.
- `log.jsonl`: Training and validation history.
- `stdout.log`: Captured execution output.
- `test_result.json`: Held-out test loss, perplexity, BPB and evaluation metadata.

Large model checkpoints (`ckpt.pt`) are stored separately in Google Drive.

## 7. Metric Definitions

**Bits per byte (BPB)** is the primary comparison metric. Lower BPB indicates better predictive compression of the evaluated text.

Unlike token-level perplexity, BPB supports comparisons between tokenizers with different vocabulary sizes and tokenization schemes.

**Tokens seen** represents the total number of training tokens processed within the fixed training-time budget, including repeated exposure when training exceeds one epoch.

**Non-embedding parameters** excludes token embeddings and positional embeddings, providing a more comparable measure of Transformer model capacity across different vocabulary sizes.

**MFU** denotes model FLOPs utilization, an estimate of achieved computational utilization relative to the hardware's theoretical capability.

## 8. Reproducibility Notes

The final experiments used a fixed budget of 1,800 seconds of optimization-step wall-clock time on a single NVIDIA Tesla T4 with FP16 mixed precision.

Preliminary learning-rate tuning used 300-second runs.

The selected learning rates are stored in `tuned_lr.json`. Two additional tuning runs at LR = 0.0064 were conducted after the original sweep. The final XS experiment retained LR = 0.0032, so the additional tuning result should not be confused with the final XS training configuration.

All test results were obtained from saved checkpoints after validation-based model selection.

The repository excludes large checkpoints and raw datasets. Reproduction of checkpoint-based evaluation requires access to the corresponding trained checkpoints and identically prepared tokenized datasets.

For implementation and execution instructions, see the [main project README](../README.md).