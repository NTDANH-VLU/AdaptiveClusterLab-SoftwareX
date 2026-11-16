"""Command-line entry point.

Use `adaptiveclusterlab demo|retail|damadics|electricity` from a clone of
the repository to reproduce the same experiments as scripts/reproduce.py.
The original `--seed / --clusters` standalone K-Means smoke test is retained
for backwards compatibility.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from .benchmark import make_example_batches, run_static_kmeans, save_metrics

DATASETS = ("demo", "retail", "damadics", "electricity")

def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in DATASETS:
        runner = Path.cwd() / "scripts" / "reproduce.py"
        if not runner.is_file():
            raise SystemExit(
                "Dataset reproduction requires a repository checkout. "
                "Run this command from its root, beside scripts/reproduce.py."
            )
        subprocess.run([sys.executable, str(runner), *sys.argv[1:]], check=True)
        return

    parser = argparse.ArgumentParser(
        description="Reference K-Means smoke test; or run: "
                    "adaptiveclusterlab {demo,retail,damadics,electricity} [options]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--output", default="results/reference_kmeans.json")
    args = parser.parse_args()
    if args.batches < 1 or args.clusters < 2:
        parser.error("--batches must be >= 1 and --clusters >= 2")
    records = run_static_kmeans(
        make_example_batches(seed=args.seed, n_batches=args.batches),
        n_clusters=args.clusters, seed=args.seed,
    )
    save_metrics(records, args.output)
    print(f"Wrote {len(records)} reference batches to {args.output}")

if __name__ == "__main__":
    main()
