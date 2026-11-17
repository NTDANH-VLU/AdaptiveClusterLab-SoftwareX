"""Regression checks for audited selection and diagnostic outputs."""
import numpy as np
from adaptiveclusterlab.controller import (
    ACTIONS, TabularQLearning, diagnostic_metrics, run_sequence, make_aba
)


def test_seeded_tie_break_not_lowest_action_bias():
    valid = [ACTIONS.index((3, 8, "reuse")), ACTIONS.index((3, 8, "update")),
             ACTIONS.index((3, 8, "retrain"))]
    decisions = []
    for seed in range(24):
        learner = TabularQLearning(seed)
        action, info = learner.choose_with_info((1, 1, 1), valid, epsilon=0)
        assert info["q_unseen_state"] and info["q_tie_break"]
        assert info["q_tie_count"] == 3
        decisions.append(action)
    assert len(set(decisions)) > 1
    learner1, learner2 = TabularQLearning(91), TabularQLearning(91)
    assert learner1.choose_with_info((1, 1, 1), valid) == learner2.choose_with_info((1, 1, 1), valid)


def test_learned_action_is_logged_without_false_tie():
    learner = TabularQLearning(0)
    state = (0, 1, 1)
    valid = [0, 1, 2]
    learner.values(state)[1] = 3.0
    action, info = learner.choose_with_info(state, valid)
    assert action == 1 and not info["q_unseen_state"]
    assert not info["q_tie_break"] and info["q_tie_count"] == 1


def test_degenerate_silhouette_is_reported_separately():
    x = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]])
    score, dbi = diagnostic_metrics(x, np.zeros(4, dtype=int))
    assert score == -1.0 and np.isnan(dbi)


def test_fixed_configuration_q_learning_is_restricted():
    sequence = make_aba(seed=101, batches=6, size=75,
                        order=["A", "A", "B", "B", "A", "A"])
    records, _ = run_sequence(sequence, seed=12, policy="q_learning_fixed",
                              development_episodes=1, development_len=3,
                              ae_epochs=1, update_epochs=1,
                              cost_mode="operation_counts")
    assert len(records) == 3
    assert all((r["k"], r["latent_dim"]) == (3, 8) for r in records)
    assert all("q_valid_values" in r and "q_unseen_state" in r for r in records)
    assert all("test_degenerate" in r and "test_silhouette_valid" in r for r in records)

def test_untrained_q_action_falls_back_to_retraining():
    learner = TabularQLearning(31)
    valid = [ACTIONS.index((3,8,"reuse")), ACTIONS.index((3,8,"retrain"))]
    choice, info = learner.choose_with_info((2,2,1), valid, require_trained=True)
    assert ACTIONS[choice] == (3, 8, "retrain")
    assert info["q_coverage_fallback"] and not info["q_action_visited"]
    assert not info["q_tie_break"]


def test_trained_q_action_is_only_candidate_when_coverage_required():
    learner = TabularQLearning(31)
    state = (2, 2, 1)
    valid = [ACTIONS.index((3,8,"reuse")), ACTIONS.index((3,8,"retrain"))]
    learner.values(state)[valid[0]] = 0.8
    learner.visits[state][valid[0]] = 3
    choice, info = learner.choose_with_info(state, valid, require_trained=True)
    assert choice == valid[0]
    assert info["q_action_visited"] and not info["q_coverage_fallback"]
