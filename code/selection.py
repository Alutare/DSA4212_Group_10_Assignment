"""Model selection helpers. Everything here uses VALIDATION results only, except `test_table`
which just collects test_result.json files that evaluate.py wrote (one test evaluation per run).

  python selection.py best      --root RUNS --stage final --study tokenizer   # -> prints best tokenizer folder
  python selection.py best      --root RUNS --stage final --study size        # -> prints best size key
  python selection.py tune_lr   --root RUNS --out ../results/tuned_lr.json    # best LR per tuned size (+ interpolation)
  python selection.py test_table --root RUNS --out ../results/final_test.csv  # val + test side by side

`best` prints exactly one token on the last stdout line so a notebook can capture it.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import experiments as E


def load_results(root, stage=None, study=None):
    out = []
    for p in sorted(Path(root).rglob("result.json")):
        r = json.loads(p.read_text())
        if stage and r.get("stage") != stage:
            continue
        if study and r.get("study") != study:
            continue
        r["_dir"] = str(p.parent)
        out.append(r)
    return out


def size_key(args):
    """Map a result's architecture back to a key of E.SIZES (None if it is a custom shape)."""
    for k, v in E.SIZES.items():
        if (args["n_layer"], args["n_head"], args["n_embd"]) == (v["n_layer"], v["n_head"], v["n_embd"]):
            return k
    return None


def usable(r):
    b = r["val"]["bpb"]
    return (not r.get("diverged")) and b == b and math.isfinite(b)


def cmd_best(a):
    rs = [r for r in load_results(a.root, a.stage, a.study) if usable(r)]
    if not rs:
        sys.exit(f"no usable results for study={a.study} stage={a.stage} in {a.root}")
    # mean over seeds of the same config, so that a multi-seed study picks by the mean
    groups = {}
    for r in rs:
        groups.setdefault(r.get("config_name", r["name"]), []).append(r)
    ranked = sorted(((np.mean([r["val"]["bpb"] for r in g]), c, g) for c, g in groups.items()), key=lambda t: t[0])
    for b, c, g in ranked:
        print(f"  {c:32s} val_bpb={b:.4f}  (n={len(g)})")
    _, cfg_name, g = ranked[0]
    if a.study in ("tokenizer", "tokenizer_control"):
        key = g[0]["tokenizer_name"]
    elif a.study == "size":
        key = size_key(g[0]["args"])
    else:
        key = cfg_name
    print(key)


def cmd_tune_lr(a):
    rs = [r for r in load_results(a.root, "tune", "lr") if usable(r)]
    best = {}
    for r in rs:
        k = size_key(r["args"])
        if k is None:
            continue
        cur = best.get(k)
        if cur is None or r["val"]["bpb"] < cur[1]:
            best[k] = (r["args"]["lr"], r["val"]["bpb"])
    if not best:
        sys.exit("no tuning runs found (run the lr study with --stage tune first)")
    grid = sorted({r["args"]["lr"] for r in rs})
    for k, (lr, b) in sorted(best.items()):
        edge = "  <-- at the edge of the grid: extend the sweep" if lr in (grid[0], grid[-1]) else ""
        print(f"  tuned {k:3s}: best lr {lr:.1e} (val bpb {b:.4f}){edge}")
    # untuned sizes: log-log interpolation of lr against non-embedding parameters (clamped at the ends)
    def n_ne(k):
        v = E.SIZES[k]
        return 12 * v["n_layer"] * v["n_embd"] ** 2
    xs = sorted(best, key=n_ne)
    lx = np.log([n_ne(k) for k in xs])
    ly = np.log([best[k][0] for k in xs])
    out = {}
    for k in E.SIZES:
        if k in best:
            out[k] = float(best[k][0])
        else:
            out[k] = float(f"{math.exp(np.interp(math.log(n_ne(k)), lx, ly)):.2g}")
    out["_note"] = ("tuned: " + ",".join(sorted(best)) + "; other sizes log-log interpolated; "
                    "tuned with 5-minute runs on validation BPB")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out, {k: v for k, v in out.items() if not k.startswith("_")})


def cmd_test_table(a):
    import pandas as pd
    rows = []
    for p in sorted(Path(a.root).rglob("test_result.json")):
        t = json.loads(p.read_text())
        rj = p.parent / "result.json"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text())
        if a.stage and r.get("stage") != a.stage:
            continue
        rows.append(dict(
            name=r["name"], config=r.get("config_name", r["name"]), study=r["study"], seed=r["seed"],
            tokenizer=r["tokenizer_name"], params_total=r["params"]["total"],
            params_nonemb=r["params"]["non_embedding"], tokens_seen=r["tokens_seen"], epochs=r["epochs"],
            tok_per_param_ne=r["tokens_per_param_nonemb"], val_bpb=r["val"]["bpb"], test_bpb=t["bpb"],
            test_loss=t["loss"], test_ppl=t["ppl"]))
    if not rows:
        sys.exit("no test_result.json found")
    df = pd.DataFrame(rows).sort_values(["study", "config", "seed"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("best")
    b.add_argument("--root", required=True)
    b.add_argument("--stage", default="final")
    b.add_argument("--study", required=True)
    b.set_defaults(fn=cmd_best)
    t = sub.add_parser("tune_lr")
    t.add_argument("--root", required=True)
    t.add_argument("--out", default=str(E.TUNED_LR_PATH))
    t.set_defaults(fn=cmd_tune_lr)
    f = sub.add_parser("test_table")
    f.add_argument("--root", required=True)
    f.add_argument("--stage", default="final")
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_test_table)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
