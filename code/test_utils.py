"""Run: python code/test_utils.py   (no torch needed)"""
import math
import numpy as np
from utils import plan_windows, bits_per_byte, lr_at, round_up, LN2


def check_plan(n, T, stride):
    plan = plan_windows(n, T, stride)
    count = np.zeros(n, dtype=int)          # how often each token index is scored
    ctx_min = []
    for s, sf in plan:
        assert 0 <= s and s + T + 1 <= n, (s, n)
        assert 0 <= sf < T
        for pos in range(sf, T):
            count[s + 1 + pos] += 1
            ctx_min.append(pos + 1)         # tokens of context the prediction sees
    assert count[0] == 0
    assert (count[1:] == 1).all(), "every token except the first must be scored exactly once"
    return plan, min(ctx_min)


for n, T, stride in [(1000, 64, 64), (1001, 64, 64), (1000, 64, 16), (999, 64, 1 + 31),
                     (65, 64, 64), (66, 64, 10), (5000, 128, 128), (5000, 128, 32)]:
    plan, cmin = check_plan(n, T, stride)
    print(f"plan ok n={n} T={T} stride={stride}: {len(plan)} windows")

# sliding window gives scored tokens more context than non-overlapping
_, c_non = check_plan(1000, 64, 64)
assert c_non == 1

# bits per byte: a model that predicts uniformly over 256 byte-tokens must score 8 bpb
n_tok = 1000
nll = n_tok * math.log(256)
assert abs(bits_per_byte(nll, n_tok, n_tok, n_tok) - 8.0) < 1e-9
# tokeniser with 4 bytes/token and loss ln(256)*4 nats/token is also 8 bpb
assert abs(bits_per_byte(4 * math.log(256) * 100, 100, 100, 400) - 8.0) < 1e-9

# LR schedule
assert abs(lr_at(0, 0.0, 1.0, 0.1, 10) - 0.1) < 1e-12
assert abs(lr_at(9, 0.0, 1.0, 0.1, 10) - 1.0) < 1e-12
assert abs(lr_at(50, 0.0, 1.0, 0.1, 10) - 1.0) < 1e-12
assert abs(lr_at(50, 1.0, 1.0, 0.1, 10) - 0.1) < 1e-12
assert abs(lr_at(50, 0.5, 1.0, 0.1, 10) - 0.55) < 1e-12
vals = [lr_at(100, p / 100, 1.0, 0.1, 10) for p in range(101)]
assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))

assert round_up(50257, 64) == 50304 and round_up(4096, 64) == 4096
print("ALL TESTS PASSED")
