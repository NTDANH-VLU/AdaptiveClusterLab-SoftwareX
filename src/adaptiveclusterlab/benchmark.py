"""Reference benchmark. No claims about Hybrid-RL or CM-HRL are made."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import davies_bouldin_score, silhouette_score

@dataclass(frozen=True)
class BatchMetrics:
    batch: int
    n_samples: int
    n_clusters: int
    silhouette: float | None
    davies_bouldin: float | None
    elapsed_seconds: float

def evaluate_labels(x: NDArray, labels: NDArray, batch: int, elapsed: float) -> BatchMetrics:
    """Score labels on the SAME input feature space for all compared models."""
    unique = np.unique(labels)
    valid = 1 < len(unique) < len(x)
    return BatchMetrics(
        batch=batch, n_samples=len(x), n_clusters=len(unique),
        silhouette=float(silhouette_score(x, labels)) if valid else None,
        davies_bouldin=float(davies_bouldin_score(x, labels)) if valid else None,
        elapsed_seconds=float(elapsed),
    )

def make_example_batches(seed: int = 42, n_batches: int = 4) -> list[NDArray]:
    """Synthetic smoke-test example, not an empirical distribution-drift study."""
    rng = np.random.default_rng(seed)
    batches = []
    for i in range(n_batches):
        x, _ = make_blobs(
            n_samples=180, n_features=4, centers=3, cluster_std=0.9,
            random_state=seed + i,
        )
        # Controlled feature offset: illustrative ONLY, not evidence of method performance.
        batches.append(np.asarray(x + rng.normal(0, 0.05, size=(1, 4)), dtype=np.float64))
    return batches

def run_static_kmeans(
    batches: list[NDArray], n_clusters: int = 3, seed: int = 42,
) -> list[BatchMetrics]:
    """Per-batch K-Means reference; NOT a frozen-model temporal baseline."""
    results = []
    for i, x in enumerate(batches):
        start = time.perf_counter()
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(x)
        elapsed = time.perf_counter() - start
        results.append(evaluate_labels(x, labels, i, elapsed))
    return results

def save_metrics(metrics: list[BatchMetrics], output: str | Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(m) for m in metrics], indent=2), encoding="utf-8")
