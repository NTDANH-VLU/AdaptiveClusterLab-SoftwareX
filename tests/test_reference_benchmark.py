import numpy as np
from adaptiveclusterlab.benchmark import (
    evaluate_labels, make_example_batches, run_static_kmeans,
)

def test_synthetic_reference_is_repeatable():
    first = make_example_batches(seed=7, n_batches=2)
    second = make_example_batches(seed=7, n_batches=2)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))

def test_reference_baseline_runs_and_is_scored():
    records = run_static_kmeans(make_example_batches(seed=9, n_batches=2))
    assert len(records) == 2
    assert all(r.n_clusters == 3 for r in records)
    assert all(r.silhouette is not None and r.davies_bouldin is not None for r in records)

def test_single_cluster_scores_are_undefined():
    x = np.array([[0.], [1.], [2.]])
    row = evaluate_labels(x, np.zeros(3, dtype=int), batch=0, elapsed=0)
    assert row.silhouette is None
    assert row.davies_bouldin is None
