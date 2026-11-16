#!/usr/bin/env python3
"""Single entry point for AdaptiveClusterLab demonstrations and public benchmarks.

Examples:
  python scripts/reproduce.py demo
  python scripts/reproduce.py retail --download
  python scripts/reproduce.py damadics --download
  python scripts/reproduce.py electricity --download
  python scripts/reproduce.py retail --input data/raw/online_retail_II.xlsx --quick
  python scripts/reproduce.py damadics --input data/raw/DAMADICS.csv --quick

Outputs are written beneath results/<dataset>/, never committed.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ELECTRICITY_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
DAMADICS_URL = (
    "https://raw.githubusercontent.com/MPiangerelli/FURAKI/main/data/DAMADICS.csv"
)


def execute(*args: str) -> None:
    print("+", sys.executable, *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        print(f"Downloading public source: {url}", flush=True)
        urllib.request.urlretrieve(url, destination)
    if destination.stat().st_size < 100:
        raise ValueError(f"Downloaded file is too small: {destination}")


def metrics_summary(csv_path: Path) -> Path:
    import numpy as np

    with csv_path.open(newline="", encoding="utf-8") as src:
        rows = list(csv.DictReader(src))
    if not rows:
        raise ValueError("No evaluation rows")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    summary = {
        "source_csv": str(csv_path.relative_to(REPO)),
        "note": "Descriptive exploratory means only; no statistically validated superiority.",
        "methods": {},
    }
    for method, items in grouped.items():
        summary["methods"][method] = {
            "n_evaluations": len(items),
            "seeds": sorted({int(x["seed"]) for x in items}),
            "test_silhouette_mean": float(
                np.mean([float(x["test_silhouette"]) for x in items])
            ),
            "test_dbi_mean": float(
                np.nanmean([float(x["test_dbi"]) for x in items])
            ),
            "reuses": sum(x["operation"] == "reuse" for x in items),
            "ae_full_fits": sum(int(x["count_ae_full"]) for x in items),
            "ae_updates": sum(int(x["count_ae_updates"]) for x in items),
            "evaluation_decision_seconds": float(
                sum(float(x["decision_seconds"]) for x in items)
            ) if "decision_seconds" in items[0] else None,
        }
    path = csv_path.with_name("summary.json")
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {path}", flush=True)
    return path


def demo(output_dir: Path, quick: bool, cost_mode: str, seeds: list[int]) -> None:
    import numpy as np
    sys.path.insert(0, str(REPO / "experiments" / "controlled"))
    from quality_cost_aba import make_aba, run_sequence

    # Small deterministic synthetic A→B→A cycle, no external dataset needed.
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in ([seeds[0]] if quick else seeds):
        sequence = make_aba(seed=seed + 100, batches=6, size=75,
                            order=["A", "A", "B", "B", "A", "A"])
        for method in ("always_retrain", "reuse_when_available",
                       "always_update", "q_learning"):
            records, _ = run_sequence(
                sequence, seed=seed, policy=method,
                development_len=3, development_episodes=1 if quick else 3,
                ae_epochs=1, update_epochs=1, cost_mode=cost_mode,
            )
            rows.extend(records)
    path = output_dir / "evaluation.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics_summary(path)


def public_data(dataset: str, supplied: Path | None, download: bool,
                output_dir: Path, quick: bool, cost_mode: str, seeds: list[int]) -> None:
    raw = REPO / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "retail":
        workbook = supplied or raw / "online_retail_II.xlsx"
        if not workbook.is_file():
            if not download:
                raise FileNotFoundError(
                    f"{workbook} not found. Use --download or --input <workbook>."
                )
            archive = raw / "online_retail_ii_uci.zip"
            fetch(RETAIL_URL, archive)
            with zipfile.ZipFile(archive) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
                if len(names) != 1:
                    raise ValueError("Expected one UCI Excel workbook")
                with z.open(names[0]) as src, workbook.open("wb") as dst:
                    while block := src.read(1 << 20):
                        dst.write(block)
        batch_dir = output_dir / "batches"
        execute("scripts/prepare_retail_ii.py", "--input", str(workbook),
                "--out", str(batch_dir), "--history-days", "90",
                "--min-customers", "15")
        prefix, months, development_months = "retail", 24, 12
    elif dataset == "electricity":
        source = supplied or raw / "LD2011_2014.txt"
        if not source.is_file():
            if not download:
                raise FileNotFoundError(f"{source} not found. Use --download or --input <txt>.")
            archive = raw / "electricity_uci.zip"
            fetch(ELECTRICITY_URL, archive)
            with zipfile.ZipFile(archive) as z:
                matches = [name for name in z.namelist() if name.endswith("LD2011_2014.txt") and not name.startswith("__MACOSX/")]
                if len(matches) != 1:
                    raise ValueError("Expected exactly one UCI Electricity text data file")
                with z.open(matches[0]) as src, source.open("wb") as dst:
                    while block := src.read(1 << 20):
                        dst.write(block)
        batch_dir = output_dir / "batches"
        execute("scripts/prepare_electricity.py", "--input", str(source),
                "--output", str(batch_dir), "--start", "2012-01",
                "--months", "24", "--development-months", "12")
        prefix, months, development_months = "electricity", 24, 12
    else:
        source = supplied or raw / "DAMADICS.csv"
        if not source.is_file():
            if not download:
                raise FileNotFoundError(
                    f"{source} not found. Use --download or --input <csv>."
                )
            fetch(DAMADICS_URL, source)
        batch_dir = output_dir / "batches"
        execute("scripts/prepare_damadics.py", "--input", str(source),
                "--output", str(batch_dir))
        prefix, months, development_months = "damadics", 8, 4
    # Identical command and policy definitions as public-data GitHub CI.
    output = output_dir / "evaluation.csv"
    execute(
        "experiments/controlled/retail_policy_runner.py",
        "--batch-dir", str(batch_dir), "--prefix", prefix, "--months", str(months),
        "--development-months", str(development_months),
        "--seeds", *([str(seeds[0])] if quick else [str(s) for s in seeds]),
        "--episodes", "1" if quick else "5",
        "--ae-epochs", "1" if quick else "2",
        "--update-epochs", "1", "--cost-mode", cost_mode,
        "--max-per-partition", "80" if quick else {"retail": "160", "damadics": "110", "electricity": "110"}[dataset],
        "--output", str(output),
    )
    metrics_summary(output)
    if dataset == "electricity":
        execute(
            "benchmarks/temporal_retail_baselines.py",
            "--batch-dir", str(batch_dir), "--prefix", "electricity",
            "--months", str(months),
            "--development-months", str(development_months),
            "--seeds", *([str(seeds[0])] if quick else [str(s) for s in seeds]),
            "--output", str(output_dir / "temporal_baselines.csv"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible AdaptiveClusterLab example or public dataset benchmark."
    )
    parser.add_argument("dataset", choices=("demo", "retail", "damadics", "electricity"))
    parser.add_argument("--input", type=Path,
                        help="Local Retail II .xlsx, DAMADICS .csv or UCI Electricity .txt")
    parser.add_argument("--download", action="store_true",
                        help="Download the specified public dataset when absent")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Independent seeds; e.g. --seeds 42 43 44 45 46")
    parser.add_argument("--cost-mode", choices=("wall_time","operation_counts"),
                        default="wall_time", help="Use operation_counts for hardware-independent controller reward; historical results used wall_time")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test budget; DO NOT use quick results in a paper")
    args = parser.parse_args()
    out = REPO / "results" / args.dataset
    if args.dataset == "demo":
        demo(out, args.quick, args.cost_mode, args.seeds)
    else:
        public_data(args.dataset, args.input, args.download, out, args.quick, args.cost_mode, args.seeds)
    print("Reproduction complete. Inspect evaluation.csv and summary.json.", flush=True)


if __name__ == "__main__":
    main()
