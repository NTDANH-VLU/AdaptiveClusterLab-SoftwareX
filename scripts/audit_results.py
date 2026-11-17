#!/usr/bin/env python3
"""Audit held-out action selection, clustering degeneracy and baseline fairness."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_finite(values):
    valid = [v for v in values if math.isfinite(v)]
    return sum(valid) / len(valid) if valid else None


def yes(value):
    return str(value).lower() in {"true", "1"}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    results = {}
    for method, items in sorted(groups.items()):
        valid = [number(r["test_silhouette_valid"]) for r in items]
        penalized = [number(r["test_silhouette"]) for r in items]
        degenerate = sum(yes(r["test_degenerate"]) for r in items)
        reused = [r for r in items if r["operation"] == "reuse"]
        results[method] = {
            "n_evaluations": len(items),
            "n_seeds": len(set(r["seed"] for r in items)),
            "valid_silhouette_mean": mean_finite(valid),
            "penalized_silhouette_mean": mean_finite(penalized),
            "n_valid_silhouette": len(items) - degenerate,
            "n_degenerate": degenerate,
            "degenerate_rate": degenerate / len(items),
            "mean_dbi_valid": mean_finite(number(r["test_dbi"]) for r in items),
            "full_fits": sum(int(r["count_ae_full"]) for r in items),
            "updates": sum(int(r["count_ae_updates"]) for r in items),
            "reuses": len(reused),
            "reuses_from_unseen_q_states": sum(yes(r["q_unseen_state"]) for r in reused),
            "reuses_from_untrained_q_actions": sum(not yes(r["q_action_visited"]) for r in reused),
            "reuses_from_trained_q_actions": sum(yes(r["q_action_visited"]) for r in reused),
            "reuses_from_tie_breaks": sum(yes(r["q_tie_break"]) for r in reused),
            "evaluation_unseen_states": sum(yes(r["q_unseen_state"]) for r in items),
            "evaluation_tie_breaks": sum(yes(r["q_tie_break"]) for r in items),
            "configurations": dict(sorted(Counter(
                f'K={r["k"]},d={r["latent_dim"]}' for r in items).items())),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.evaluation_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Evaluation CSV is empty")
    needed = {"method","seed","test_silhouette","test_silhouette_valid",
              "test_degenerate","q_unseen_state","q_tie_break","q_action_visited","k",
              "latent_dim","count_ae_full","count_ae_updates","operation"}
    assert needed.issubset(rows[0]), f"Missing audit fields: {needed-set(rows[0])}"
    result = {"source": str(args.evaluation_csv), "methods": summarize(rows)}
    output = args.output or args.evaluation_csv.with_name("audit_report.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["methods"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
