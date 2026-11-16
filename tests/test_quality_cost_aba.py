"""Runtime invariants for experimental quality–cost controller (not paper proof)."""
from pathlib import Path
import sys
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "controlled"))
from quality_cost_aba import (
    ACTIONS, PolicyEnvironment, TabularQLearning,
    make_aba, split_three, run_sequence,
)


def observations():
    return make_aba(seed=101, batches=6, size=75,
                    order=["A", "A", "B", "B", "A", "A"])


def test_actual_reuse_update_retrain_cost_counters_and_shapes():
    x = observations()[0][1]
    tr, val, test = split_three(x, 45)
    env = PolicyEnvironment(seed=9, ae_epochs=1, update_epochs=1)
    fresh = ACTIONS.index((3, 8, "retrain"))
    reuse = ACTIONS.index((3, 8, "reuse"))
    update = ACTIONS.index((3, 8, "update"))
    assert fresh in env.possible()
    assert reuse not in env.possible() and update not in env.possible()
    a = env.act(tr, val, test, fresh, regime="A")
    assert a["count_ae_full"] == 1 and a["count_kmeans_full"] == 1
    assert a["latent_dim"] == 8
    assert reuse in env.possible() and update in env.possible()
    b = env.act(tr, val, test, reuse, regime="A")
    assert b["count_ae_full"] == 0 and b["count_ae_updates"] == 0
    assert b["count_kmeans_full"] == 0 and b["count_kmeans_updates"] == 0
    assert b["count_reuse"] == 1
    c = env.act(tr, val, test, update, regime="A")
    assert c["count_ae_updates"] == 1 and c["count_kmeans_updates"] == 1
    assert c["count_ae_full"] == 0 and c["count_kmeans_full"] == 0
    assert env.bank[-1].latent_centers.shape == (3, 8)
    assert env.bank[-1].common_centers.shape == (3, 5)
    assert all(np.isfinite(r["test_silhouette"]) for r in (a, b, c))


def test_recurring_regime_has_distinct_B_distribution():
    s = observations()
    mean_a = np.mean(s[0][1], axis=0)
    mean_b = np.mean(s[2][1], axis=0)
    mean_a_return = np.mean(s[4][1], axis=0)
    assert np.linalg.norm(mean_a - mean_a_return) < np.linalg.norm(mean_a - mean_b)


def test_q_policy_updates_and_reports_heldout_independently():
    records, totals = run_sequence(
        observations(), seed=12, policy="q_learning",
        development_episodes=2, development_len=3,
        ae_epochs=1, update_epochs=1, cost_weight=.01)
    assert len(records) == 3
    assert all(r["policy_training_updates"] == 6 for r in records)
    assert all(r["end_to_end_decision_seconds"] >= r["total_seconds"] for r in records)
    assert all(r["policy_training_seconds"] > 0 for r in records)
    assert all(r["memory_warmup_seconds"] > 0 for r in records)
    assert sum(totals.values()) >= 3
    assert set(r["operation"] for r in records).issubset({"reuse", "update", "retrain"})
    assert all(np.isfinite(r["test_silhouette"]) for r in records)


def test_controlled_baselines_have_expected_actions():
    seq = observations()
    a, ta = run_sequence(seq, seed=12, policy="always_retrain",
                         development_len=3, ae_epochs=1)
    b, tb = run_sequence(seq, seed=12, policy="reuse_when_available",
                         development_len=3, ae_epochs=1)
    assert all(row["operation"] == "retrain" for row in a)
    assert all(row["operation"] == "reuse" for row in b)
    assert ta["ae_full"] == 3
    assert tb["ae_full"] == 0


def test_returning_A_regime_can_retrieve_matching_A_snapshot():
    seq = observations()
    env = PolicyEnvironment(seed=42, ae_epochs=1, update_epochs=1)
    for index in (0, 2):
        tr, val, test = split_three(seq[index][1], seed=12 + index)
        env.act(tr, val, test, ACTIONS.index((3, 8, "retrain")),
                regime=seq[index][2])
    return_train, _, _ = split_three(seq[4][1], seed=16)
    best, distance = env.nearest(return_train, 3, 8)
    assert best is not None
    assert best.regime == "A"
    assert np.isfinite(distance)


def test_reuse_gate_blocks_mismatched_regime_before_decision():
    seq = observations()
    env = PolicyEnvironment(seed=42, ae_epochs=1, reuse_threshold=0.65)
    a_train, a_val, a_test = split_three(seq[0][1], 12)
    env.act(a_train, a_val, a_test, ACTIONS.index((3, 8, "retrain")), regime="A")
    b_train, _, _ = split_three(seq[2][1], 14)
    return_train, _, _ = split_three(seq[4][1], 16)
    reuse = ACTIONS.index((3, 8, "reuse"))
    update = ACTIONS.index((3, 8, "update"))
    assert reuse not in env.possible(b_train)
    assert reuse in env.possible(return_train)
    assert update in env.possible(b_train)


def test_gated_q_controller_is_separate_ablation():
    records, _ = run_sequence(
        observations(), seed=12, policy="q_learning_gated",
        development_episodes=1, development_len=3,
        ae_epochs=1, update_epochs=1, cost_weight=.01)
    assert len(records) == 3
    assert all(row["method"] == "q_learning_gated" for row in records)


def test_deterministic_cost_proxy_executes_without_wall_clock_reward():
    seq = observations()
    records, counts = run_sequence(
        seq, seed=12, policy="q_learning",
        development_episodes=1, development_len=3,
        ae_epochs=1, update_epochs=1, cost_weight=.02,
        cost_mode="operation_counts",
    )
    assert len(records) == 3
    assert sum(counts.values()) >= 3
    assert all(np.isfinite(row["test_silhouette"]) for row in records)


def test_invalid_cost_mode_rejected():
    with pytest.raises(ValueError, match="cost_mode"):
        run_sequence(observations(), development_len=3, cost_mode="invalid")
