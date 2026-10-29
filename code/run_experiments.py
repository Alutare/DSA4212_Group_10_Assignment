"""Run a study (sequentially, resumable, skips finished runs).

Examples
  python run_experiments.py --study tokenizer --stage final --dry_run
  python run_experiments.py --study lr --stage tune --size m --tok bpe16k
  python run_experiments.py --study size --stage final --tok bpe16k
  python run_experiments.py --study seeds --stage final --size m --tok bpe16k --seeds 1337 1338 1339
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import experiments as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=E.STUDIES)
    ap.add_argument("--stage", default="final", choices=["tune", "final", "smoke"])
    ap.add_argument("--tok", default="gpt2")
    ap.add_argument("--size", default="m")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337])
    ap.add_argument("--data_root", default="/content/data")
    ap.add_argument("--out_root", default="/content/drive/MyDrive/DSA4212/runs")
    ap.add_argument("--budget", type=float, default=None, help="override seconds (default: declared budget)")
    ap.add_argument("--only", default="", help="substring filter on config name")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    budget = a.budget or (E.BUDGET_SECONDS * (E.TUNE_FRACTION if a.stage == "tune" else 1.0))
    if a.stage == "smoke":
        budget = a.budget or 60
    here = Path(__file__).parent
    jobs = []
    for run in E.build_runs(a.study, a.tok, a.size):
        if a.only and a.only not in run["config"]:
            continue
        for seed in a.seeds:
            name = run["config"] + (f"_s{seed}" if len(a.seeds) > 1 or seed != 1337 else "")
            out_dir = Path(a.out_root) / a.stage / a.study / name
            cmd = [sys.executable, str(here / "train.py"), "--name", name, "--config_name", run["config"],
                   "--study", a.study, "--stage", a.stage, "--seed", str(seed),
                   "--data_dir", str(Path(a.data_root) / run["tok"]), "--out_dir", str(out_dir),
                   "--budget_seconds", str(budget)]
            for k, v in run["args"].items():
                if isinstance(v, bool):
                    if v:
                        cmd.append(f"--{k}")
                else:
                    cmd += [f"--{k}", str(v)]
            jobs.append((name, out_dir, cmd))

    print(f"{len(jobs)} run(s), {budget:.0f}s budget each -> ~{len(jobs) * budget / 3600:.1f} h of training")
    failed = []
    for i, (name, out_dir, cmd) in enumerate(jobs, 1):
        done = (out_dir / "result.json").exists()
        print(f"\n[{i}/{len(jobs)}] {name} {'(done, will skip)' if done else ''}")
        if a.dry_run:
            print("   ", " ".join(cmd))
            continue
        if done:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(out_dir / "stdout.log", "a") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        tail = (out_dir / "stdout.log").read_text().splitlines()[-2:]
        print(f"    exit={rc}, {time.time() - t0:.0f}s wall | " + " | ".join(tail))
        if rc != 0:
            failed.append(name)
    if failed:
        print("\nFAILED:", failed, "(re-run the same command to resume them)")


if __name__ == "__main__":
    main()
