"""Torch-free helpers (unit-tested in test_utils.py)."""
import json
import math
import os
import subprocess
from pathlib import Path

LN2 = math.log(2.0)


def plan_windows(n_tokens, block_size, stride):
    """Plan evaluation windows over a token stream.

    Returns a list of (start, score_from). For a window starting at `start`,
    inputs are data[start:start+T] and targets are data[start+1:start+T+1].
    Only positions >= score_from are scored, so that EVERY token except the
    very first one (which has no context) is scored exactly once.
    stride == block_size -> non-overlapping windows (cheap, standard).
    stride <  block_size -> sliding window; scored tokens get more context.
    """
    T = block_size
    if not (1 <= stride <= T):
        raise ValueError("need 1 <= stride <= block_size")
    if n_tokens < T + 1:
        raise ValueError("stream shorter than block_size + 1")
    plan, covered_end, s = [], 0, 0
    while True:
        if s + T + 1 > n_tokens:          # last, right-aligned window
            s = n_tokens - 1 - T
        end = s + T                        # index of last target token
        first_new = max(covered_end + 1, s + 1)
        plan.append((s, first_new - (s + 1)))
        covered_end = end
        if end >= n_tokens - 1:
            break
        s += stride
    return plan


def bits_per_byte(nll_sum, n_scored, n_tokens_total, n_bytes_total):
    """Tokeniser-independent metric.

    nll_sum/n_scored = mean nats per token over scored tokens. Multiplying by
    (tokens in split / UTF-8 bytes in split) converts to nats per byte, and
    dividing by ln 2 gives bits per byte.
    """
    return (nll_sum / n_scored) * (n_tokens_total / n_bytes_total) / LN2


def lr_at(step, progress, lr, min_lr, warmup_steps):
    """Linear warm-up (in steps), then cosine decay driven by the fraction of
    the TIME budget already used (progress in [0, 1])."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    p = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * p)) * (lr - min_lr)


def round_up(v, m):
    return ((v + m - 1) // m) * m


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def git_sha(cwd=None):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None
