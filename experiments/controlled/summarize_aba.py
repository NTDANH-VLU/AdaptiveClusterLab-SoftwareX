"""Summarize controlled A-B-A results without overstating statistical evidence.

Reads exact executable output CSV. Computes per-seed paired comparisons.
Do not treat within-seed months as independent statistical replicates.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def summarize(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    groups=defaultdict(list)
    for row in rows:
        groups[(int(row["seed"]),row["method"])].append(row)
    statistics={}
    for (seed,method),sample in groups.items():
        routing_included = sum(float(r["end_to_end_decision_seconds"]) for r in sample)
        development_seconds = float(sample[0]["policy_training_seconds"])
        memory_warmup_seconds = float(sample[0]["memory_warmup_seconds"])
        statistics[(seed,method)]={
            "seed":seed,"method":method,"batches":len(sample),
            "test_silhouette":float(np.mean([float(r["test_silhouette"]) for r in sample])),
            "test_dbi":float(np.nanmean([float(r["test_dbi"]) for r in sample])),
            "cost_seconds":routing_included,
            "policy_training_seconds":development_seconds,
            "memory_warmup_seconds":memory_warmup_seconds,
            "total_including_development_and_warmup_seconds":
                routing_included + development_seconds + memory_warmup_seconds,
            "full_ae_fits":sum(int(r["count_ae_full"]) for r in sample),
            "ae_updates":sum(int(r["count_ae_updates"]) for r in sample),
            "cluster_full_fits":sum(int(r["count_kmeans_full"]) for r in sample),
            "cluster_updates":sum(int(r["count_kmeans_updates"]) for r in sample),
            "reuse":sum(r["operation"]=="reuse" for r in sample),
            "recovered_A_reuses":sum(r["operation"]=="reuse" and
                                    r["true_regime_for_audit"]=="A" and
                                    r["selected_memory_regime"]=="A" for r in sample),
            "negative_transfer_B_from_A":sum(r["operation"]=="reuse" and
                   r["true_regime_for_audit"]=="B" and
                   r["selected_memory_regime"]=="A" for r in sample),
        }
    paired=[]
    for seed in sorted({key[0] for key in statistics}):
        base=statistics.get((seed,"always_retrain"))
        proposed=statistics.get((seed,"q_learning"))
        if not base or not proposed:
            raise ValueError("Missing paired always_retrain or q_learning row")
        if base["batches"]!=proposed["batches"]:
            raise ValueError("Different evaluation batch count")
        paired.append(dict(
            seed=seed,
            delta_silhouette=proposed["test_silhouette"]-base["test_silhouette"],
            total_cost_reduction=(base["cost_seconds"]-proposed["cost_seconds"])/base["cost_seconds"]
                                  if base["cost_seconds"]>0 else float("nan"),
            total_cost_reduction_including_development=(
                (base["total_including_development_and_warmup_seconds"] -
                 proposed["total_including_development_and_warmup_seconds"]) /
                 base["total_including_development_and_warmup_seconds"]
                if base["total_including_development_and_warmup_seconds"] > 0 else float("nan")
            ),
            avoided_full_ae_fits=base["full_ae_fits"]-proposed["full_ae_fits"],
            additional_ae_updates=proposed["ae_updates"]-base["ae_updates"],
        ))
    return list(statistics.values()),paired

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    results,paired=summarize(args.input)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    payload={
        "per_seed_method":results,
        "paired_q_vs_full_retrain":paired,
        "unweighted_mean_paired_delta_silhouette":float(np.mean(
            [p["delta_silhouette"] for p in paired])),
        "unweighted_mean_paired_cost_reduction":float(np.mean(
            [p["total_cost_reduction"] for p in paired])),
        "warning":"Evaluation costs include action-routing overhead, while separate total cost includes Q development and memory warmup. Small pilot; evaluate independent seeds, held-out quality, and full cost. "
                  "No significance or universal superiority claim.",
    }
    output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload["paired_q_vs_full_retrain"],indent=2))
    print("Saved",output)

if __name__=="__main__":main()
