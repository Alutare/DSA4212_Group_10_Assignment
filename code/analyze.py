"""Collect result.json files -> summary tables + report figures.

python analyze.py --root /content/drive/MyDrive/DSA4212/runs --stage final --out figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect(root, stage):
    rows, curves = [], {}
    for p in sorted(Path(root).rglob("result.json")):
        r = json.loads(p.read_text())
        if stage and r.get("stage") != stage:
            continue
        a, v, pc = r["args"], r["val"], r["params"]
        rows.append(dict(
            name=r["name"], config=r.get("config_name", r["name"]), study=r["study"], seed=r["seed"],
            tokenizer=r["tokenizer_name"], vocab=r["vocab_size"], n_layer=a["n_layer"], n_head=a["n_head"],
            n_embd=a["n_embd"], block_size=a["block_size"], lr=a["lr"], tokens_per_step=r["tokens_per_step"],
            dtype=a["dtype"], compiled=a.get("compile", False), train_fraction=a.get("train_fraction", 1.0),
            params_total=pc["total"], params_nonemb=pc["non_embedding"], tokens_seen=r["tokens_seen"],
            epochs=r["epochs"], tok_per_param_ne=r["tokens_per_param_nonemb"], tok_s=r["tokens_per_sec"],
            mfu=r["mfu"], peak_mem_gb=r["peak_mem_gb"], train_time=r["train_time"], val_loss=v["loss"],
            val_ppl=v["ppl"], val_bpb=v["bpb"], diverged=r["diverged"], resumes=r["n_resumes"], dir=str(p.parent)))
        curves[r["name"]] = r["val_curve"]
    return pd.DataFrame(rows), curves


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(f"{x:.4g}" if isinstance(x, (float, np.floating)) else str(x)
                                     for x in row) + " |")
    return "\n".join(out)


def fig_save(fig, out, name):
    fig.tight_layout()
    fig.savefig(Path(out) / f"{name}.png", dpi=170)
    plt.close(fig)


def agg(df):
    g = df.groupby(["study", "config"], as_index=False).agg(
        n=("val_bpb", "size"), val_bpb=("val_bpb", "mean"), bpb_std=("val_bpb", "std"),
        val_ppl=("val_ppl", "mean"), params_nonemb=("params_nonemb", "first"),
        params_total=("params_total", "first"), vocab=("vocab", "first"), n_layer=("n_layer", "first"),
        n_embd=("n_embd", "first"), block_size=("block_size", "first"), lr=("lr", "first"),
        tokens_per_step=("tokens_per_step", "first"), train_fraction=("train_fraction", "first"),
        tokens_seen=("tokens_seen", "mean"), epochs=("epochs", "mean"), tok_per_param_ne=("tok_per_param_ne", "mean"),
        tok_s=("tok_s", "mean"), mfu=("mfu", "mean"))
    g["bpb_std"] = g["bpb_std"].fillna(0.0)
    return g


def curves_plot(df, curves, study, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for _, r in df[df.study == study].iterrows():
        c = pd.DataFrame(curves[r["name"]]).drop_duplicates("step", keep="last")
        c = c[c.step > 0]
        ax[0].plot(c.train_time / 60, c.bpb, label=r["config"])
        ax[1].plot(c.tokens / 1e6, c.bpb, label=r["config"])
    ax[0].set_xlabel("training time (min)")
    ax[1].set_xlabel("training tokens seen (M)")
    for x in ax:
        x.set_ylabel("validation bits/byte")
        x.set_yscale("log")
        x.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"{study}: validation BPB vs time")
    ax[1].set_title(f"{study}: validation BPB vs tokens")
    fig_save(fig, out, f"{study}_curves")


def line_with_err(ax, x, y, e, label=None, log_x=False, annotate=None):
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=label)
    if log_x:
        ax.set_xscale("log")
    if annotate is not None:
        for xi, yi, t in zip(x, y, annotate):
            ax.annotate(str(t), (xi, yi), textcoords="offset points", xytext=(4, 5), fontsize=7)
    ax.grid(alpha=.3)
    ax.set_ylabel("validation bits/byte (lower = better)")


def study_plots(g, df, curves, study, out):
    d = g[g.study == study].copy()
    if d.empty:
        return
    if study == "tokenizer":
        d = d.sort_values("vocab")
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        line_with_err(ax[0], d.vocab, d.val_bpb, d.bpb_std, log_x=True, annotate=d.config)
        ax[0].set_xlabel("vocabulary size")
        ax[1].plot(d.vocab, d.tok_s, marker="o")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("vocabulary size")
        ax[1].set_ylabel("tokens / s")
        ax[1].grid(alpha=.3)
        ax[2].plot(d.vocab, d.params_nonemb / d.params_total, marker="o")
        ax[2].set_xscale("log")
        ax[2].set_xlabel("vocabulary size")
        ax[2].set_ylabel("non-embedding / total params")
        ax[2].grid(alpha=.3)
    elif study == "size":
        d = d.sort_values("params_nonemb")
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        line_with_err(ax[0], d.params_nonemb, d.val_bpb, d.bpb_std, log_x=True, annotate=d.config)
        ax[0].set_xlabel("non-embedding parameters")
        ax[1].plot(d.params_nonemb, d.tokens_seen / 1e6, marker="o")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("non-embedding parameters")
        ax[1].set_ylabel("tokens seen in budget (M)")
        ax[1].grid(alpha=.3)
        d3 = d.sort_values("tok_per_param_ne")
        ax[2].plot(d3.tok_per_param_ne, d3.val_bpb, marker="o")
        ax[2].set_xscale("log")
        ax[2].set_xlabel("tokens per non-embedding parameter")
        ax[2].set_ylabel("validation bits/byte")
        ax[2].grid(alpha=.3)
    else:
        xcol, xlabel, logx = {
            "context": ("block_size", "context length", True), "lr": ("lr", "peak learning rate", True),
            "batch": ("tokens_per_step", "tokens per optimiser step", True),
            "data": ("train_fraction", "fraction of training corpus", True)}.get(study, (None, None, False))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        if xcol:
            d = d.sort_values(xcol)
            line_with_err(ax[0], d[xcol], d.val_bpb, d.bpb_std, log_x=logx)
            ax[0].set_xlabel(xlabel)
        else:
            ax[0].bar(d.config, d.val_bpb, yerr=d.bpb_std, capsize=3)
            ax[0].set_ylabel("validation bits/byte")
            ax[0].tick_params(axis="x", rotation=30, labelsize=7)
        ax[1].bar(d.config, d.tok_s)
        ax[1].set_ylabel("tokens / s")
        ax[1].tick_params(axis="x", rotation=30, labelsize=7)
        if study == "data":
            ax[1].clear()
            ax[1].plot(d.epochs, d.val_bpb, marker="o")
            ax[1].set_xscale("log")
            ax[1].set_xlabel("epochs over the (sub)corpus")
            ax[1].set_ylabel("validation bits/byte")
            ax[1].grid(alpha=.3)
    fig.suptitle(f"Study: {study}")
    fig_save(fig, out, f"{study}_summary")
    curves_plot(df, curves, study, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", default="final")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df, curves = collect(a.root, a.stage)
    if df.empty:
        raise SystemExit("no result.json files found")
    df.sort_values(["study", "config", "seed"]).to_csv(Path(a.out) / "all_runs.csv", index=False)
    g = agg(df)
    g.to_csv(Path(a.out) / "summary_by_config.csv", index=False)
    (Path(a.out) / "summary_by_config.md").write_text(
        md_table(g.sort_values(["study", "val_bpb"])[["study", "config", "n", "params_nonemb", "params_total",
                                                       "tokens_seen", "epochs", "tok_per_param_ne", "tok_s", "mfu",
                                                       "val_bpb", "bpb_std", "val_ppl"]]))
    for study in sorted(df.study.unique()):
        study_plots(g, df, curves, study, a.out)
    print(g.sort_values(["study", "val_bpb"])[["study", "config", "n", "params_nonemb", "tokens_seen",
                                                "epochs", "val_bpb", "bpb_std"]].to_string(index=False))
    if (df.diverged).any():
        print("\nWARNING diverged runs:", df[df.diverged].name.tolist())
    print("\nfigures + tables in", a.out)


if __name__ == "__main__":
    main()
