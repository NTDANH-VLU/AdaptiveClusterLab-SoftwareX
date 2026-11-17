"""A testable quality–cost RL policy: choose (K, d, reuse/update/retrain).

EXPERIMENTAL research variant, not the published Hybrid-RL or CM-HRL.
Uses the author's original ClusteringAE and training function from legacy_cmhrl.
The control policy is tabular Q-learning, not the original CM-HRL neural policy.

Training policy uses developmental sequences and validation splits only.
Evaluation policy is frozen. Test subsets NEVER select actions or control gates.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Legacy architecture remains unchanged; no notebook is modified.
from . import _ae_bridge as legacy

ACTIONS = tuple(
    (k, dim, operation)
    for k in (3, 5)
    for dim in (8, 12)
    for operation in ("reuse", "update", "retrain")
)
OPERATIONS = ("reuse", "update", "retrain")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    legacy.device = torch.device("cpu")


def silhouette(x, labels) -> float:
    if not 1 < len(np.unique(labels)) < len(x):
        return -1.0
    return float(silhouette_score(x, labels))


def diagnostic_metrics(x, labels) -> tuple[float, float]:
    score = silhouette(x, labels)
    dbi = float(davies_bouldin_score(x, labels)) if score > -1 else float("nan")
    return score, dbi


def split_three(x: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    """Independent observation splits (60/20/20), no test feedback."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    n_train = int(len(x) * 0.6)
    n_val = int(len(x) * 0.2)
    return (x[indices[:n_train]], x[indices[n_train:n_train+n_val]],
            x[indices[n_train+n_val:]])


def latent(model, x: np.ndarray) -> np.ndarray:
    z = legacy.get_latent(model, np.asarray(x, dtype=np.float32))
    return np.asarray(z, dtype=np.float64)


def assign(z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)


def input_prototypes(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    # Guarantee one 5D prototype per cluster; empty cluster gets global mean.
    return np.stack([x[labels == i].mean(axis=0) if np.any(labels == i)
                     else x.mean(axis=0) for i in range(k)]).astype(np.float32)


@dataclass
class Snapshot:
    k: int
    dim: int
    encoder_state: dict
    latent_centers: np.ndarray
    common_centers: np.ndarray
    train_mean: np.ndarray
    regime: str


class PolicyEnvironment:
    """Memory entries hold matched AE weights + latent/common-space prototypes."""

    def __init__(self, ae_epochs=2, update_epochs=1, memory_size=24, seed=42, reuse_threshold=None):
        self.ae_epochs = int(ae_epochs)
        self.update_epochs = int(update_epochs)
        self.memory_size = int(memory_size)
        self.seed = int(seed)
        self.reuse_threshold = reuse_threshold
        self.bank: list[Snapshot] = []
        self.last_mean = None
        self.last_regime = ""
        self.counters = dict(ae_full=0, ae_updates=0, kmeans_full=0,
                             kmeans_updates=0, reuse=0)

    def nearest(self, x_train: np.ndarray, k: int, dim: int):
        candidates = [s for s in self.bank if (s.k, s.dim) == (k, dim)]
        if not candidates:
            return None, float("inf")
        mean = x_train.mean(axis=0)
        scores = [float(np.linalg.norm(mean - s.train_mean)) for s in candidates]
        idx = int(np.argmin(scores))
        return candidates[idx], scores[idx]

    def possible(self, x_train=None) -> list[int]:
        allowed = []
        for i, (k, dim, op) in enumerate(ACTIONS):
            if op == "retrain":
                allowed.append(i)
                continue
            candidate, distance = self.nearest(x_train, k, dim) if x_train is not None else (None, float("inf"))
            if x_train is None:
                candidate = next((s for s in self.bank if (s.k, s.dim) == (k, dim)), None)
            if candidate is None:
                continue
            if op == "reuse" and self.reuse_threshold is not None and (
                x_train is None or distance > self.reuse_threshold
            ):
                continue
            allowed.append(i)
        return allowed

    def state(self, x_train: np.ndarray) -> tuple[int, int, int]:
        if self.last_mean is None:
            drift = 2
        else:
            dist = float(np.linalg.norm(x_train.mean(axis=0) - self.last_mean))
            drift = 0 if dist < 0.4 else 1 if dist < 1.2 else 2
        matches = [self.nearest(x_train, k, dim)[1] for k in (3, 5) for dim in (8, 12)]
        nearest = min(matches)
        memory_sim = 2 if not np.isfinite(nearest) else (0 if nearest < 0.4 else 1 if nearest < 1.2 else 2)
        return drift, memory_sim, int(bool(self.bank))

    def _build(self, x, dim, epochs, seed, existing=None):
        seed_all(seed)
        model = legacy.ClusteringAE(in_dim=x.shape[1], latent=dim).to(legacy.device)
        if existing is not None:
            model.load_state_dict(copy.deepcopy(existing.encoder_state))
        legacy.train_clustering_ae(np.asarray(x, dtype=np.float32), model,
                                   epochs=epochs, alpha=0.0, lr=1e-3)
        return model

    def _restore(self, x, snapshot):
        model = legacy.ClusteringAE(in_dim=x.shape[1], latent=snapshot.dim).to(legacy.device)
        model.load_state_dict(copy.deepcopy(snapshot.encoder_state))
        return model

    def act(self, x_train, x_validation, x_test, action, regime="unknown"):
        """Execute action, return data-independent cost counters and three-way scores.

        Important: x_test is evaluated ONLY after model decision; no test metric
        enters the next policy state, reward, or candidate selection.
        """
        k, dim, op = ACTIONS[action]
        selected, distance = self.nearest(x_train, k, dim)
        if op != "retrain" and selected is None:
            raise ValueError("Cannot update/reuse without matching memory")
        before = self.counters.copy()
        t0 = time.perf_counter()
        ae_train_sec = cluster_train_sec = 0.0
        if op == "retrain":
            start = time.perf_counter()
            model = self._build(x_train, dim, self.ae_epochs, self.seed + k * 31 + dim)
            ae_train_sec = time.perf_counter() - start
            z = latent(model, x_train)
            start = time.perf_counter()
            centers = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(z).cluster_centers_
            cluster_train_sec = time.perf_counter() - start
            self.counters["ae_full"] += 1
            self.counters["kmeans_full"] += 1
        elif op == "reuse":
            model = self._restore(x_train, selected)
            centers = selected.latent_centers.copy()
            self.counters["reuse"] += 1
        else:
            start = time.perf_counter()
            model = self._build(x_train, dim, self.update_epochs,
                                self.seed + k * 31 + dim, existing=selected)
            ae_train_sec = time.perf_counter() - start
            # Re-embed archived common-space prototypes with the updated encoder:
            # do not mix latent coordinates from two different encoders.
            transformed = latent(model, selected.common_centers)
            z = latent(model, x_train)
            start = time.perf_counter()
            centers = MiniBatchKMeans(n_clusters=k, init=transformed,
                n_init=1, max_iter=5, batch_size=min(len(z), 128),
                random_state=self.seed).fit(z).cluster_centers_
            cluster_train_sec = time.perf_counter() - start
            self.counters["ae_updates"] += 1
            self.counters["kmeans_updates"] += 1
        assert centers.shape == (k, dim)
        z_train = latent(model, x_train)
        train_labels = assign(z_train, centers)
        new_snapshot = Snapshot(
            k=k, dim=dim, encoder_state=copy.deepcopy(model.state_dict()),
            latent_centers=centers.copy(),
            common_centers=input_prototypes(x_train, train_labels, k),
            train_mean=x_train.mean(axis=0).copy(), regime=regime,
        )
        # Store only models created or updated, not every read-only reuse.
        if op != "reuse":
            self.bank.append(new_snapshot)
            self.bank = self.bank[-self.memory_size:]
        self.last_mean = x_train.mean(axis=0).copy()
        self.last_regime = regime
        z_val = latent(model, x_validation)
        val_sil, val_dbi = diagnostic_metrics(x_validation, assign(z_val, centers))
        elapsed = time.perf_counter() - t0  # excludes final TEST evaluation cost
        z_test = latent(model, x_test)
        test_sil, test_dbi = diagnostic_metrics(x_test, assign(z_test, centers))
        return dict(operation=op, k=k, latent_dim=dim,
            val_silhouette_valid=val_sil if val_sil > -1 else float("nan"),
            test_silhouette_valid=test_sil if test_sil > -1 else float("nan"),
            val_degenerate=bool(val_sil == -1),
            test_degenerate=bool(test_sil == -1),
            selected_memory_regime=selected.regime if selected is not None else "",
            memory_distance=float(distance) if selected is not None else None,
            val_silhouette=val_sil, val_dbi=val_dbi,
            test_silhouette=test_sil, test_dbi=test_dbi,
            total_seconds=elapsed, ae_training_seconds=ae_train_sec,
            clustering_seconds=cluster_train_sec, **{
                "count_" + field: self.counters[field] - before[field] for field in before
            })


class TabularQLearning:
    """Small but genuine online RL controller over 12 config/mode actions."""

    def __init__(self, seed=42, learning_rate=0.25, gamma=0.85):
        self.q = {}
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def values(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(len(ACTIONS), dtype=np.float64)
        return self.q[state]

    def choose_with_info(self, state, valid, epsilon=0.0):
        """Return the chosen action and an auditable diagnostic of the decision.

        Q-values for unseen states start at zero. Ties are broken with the
        seeded RNG rather than selecting the lowest action index.
        """
        if not valid:
            raise ValueError("No valid actions")
        unseen = state not in self.q
        values = self.values(state)
        best = max(float(values[i]) for i in valid)
        tied = [i for i in valid if np.isclose(values[i], best, rtol=0, atol=1e-12)]
        exploratory = bool(self.rng.random() < epsilon)
        action = int(self.rng.choice(valid if exploratory else tied))
        ranked = sorted((float(values[i]) for i in valid), reverse=True)
        return action, dict(
            q_state=str(state), q_unseen_state=bool(unseen),
            q_tie_count=len(tied), q_tie_break=bool(not exploratory and len(tied)>1),
            q_exploration=exploratory, q_selected_value=float(values[action]),
            q_best_value=best,
            q_margin=float(ranked[0]-ranked[1]) if len(ranked)>1 else float("nan"),
            q_valid_values=json.dumps({str(i):float(values[i]) for i in valid}),
        )

    def choose(self, state, valid, epsilon=0.0):
        action, _ = self.choose_with_info(state, valid, epsilon=epsilon)
        return action

    def learn(self, state, action, reward, next_state, next_valid):
        current = self.values(state)
        future = max(self.values(next_state)[i] for i in next_valid)
        current[action] += self.lr * (
            reward + self.gamma * future - current[action]
        )
        self.updates += 1


def make_aba(seed=42, batches=12, size=150, order=None):
    """Recurring but genuinely DIFFERENT A/B regimes, labels for audit ONLY."""
    rng = np.random.default_rng(seed)
    if order is None:
        order = ["A"] * (batches // 3) + ["B"] * (batches // 3)
        order += ["A"] * (batches - len(order))
    observations = []
    for t, regime in enumerate(order):
        centers = (np.array([[-1.4, -1.0, 0., 0., 0.],
                             [0.3, 1.4, -0.8, .5, .2],
                             [1.5, -0.7, 1., -.4, .7]])
                   if regime == "A" else
                   np.array([[2.0, 2.0, 1.2, -.6, .7],
                             [-1.5, 1.8, 1.8, .4, -.7],
                             [1.1, -2., -1., 1.6, 1.5]]))
        components = rng.integers(0, 3, size=size)
        features = centers[components] + rng.normal(0, .4, size=(size, 5))
        observations.append((f"{t:02d}", features.astype(np.float32), regime))
    return observations


def fixed_action(valid, policy):
    if policy == "always_retrain":
        return next(i for i in valid if ACTIONS[i] == (3, 8, "retrain"))
    if policy == "reuse_when_available":
        reuse = next((i for i in valid if ACTIONS[i] == (3, 8, "reuse")), None)
        return reuse if reuse is not None else fixed_action(valid, "always_retrain")
    if policy == "always_update":
        update = next((i for i in valid if ACTIONS[i] == (3, 8, "update")), None)
        return update if update is not None else fixed_action(valid, "always_retrain")
    raise ValueError(policy)


def run_sequence(sequence, seed=42, policy="q_learning",
                 development_episodes=5, development_len=6,
                 ae_epochs=2, update_epochs=1, cost_weight=.02,
                 cost_mode="wall_time"):
    if development_len >= len(sequence):
        raise ValueError("Need held-out evaluation sequence after development")
    if cost_weight < 0:
        raise ValueError("cost_weight must be nonnegative")
    if cost_mode not in ("wall_time", "operation_counts"):
        raise ValueError("cost_mode must be wall_time or operation_counts")
    development_start = time.perf_counter()
    learner = TabularQLearning(seed)
    if policy in ("q_learning", "q_learning_gated", "q_learning_fixed"):
        # Repeated development scenarios; no evaluation observation is used.
        # Different stochastic splits per episode, policy persists across episodes.
        for episode in range(development_episodes):
            environment = PolicyEnvironment(
                ae_epochs, update_epochs, seed=seed,
                reuse_threshold=0.65 if policy == "q_learning_gated" else None
            )
            for t, (_, x, regime) in enumerate(sequence[:development_len]):
                train, val, test = split_three(x, seed + episode * 1000 + t)
                state = environment.state(train)
                valid = environment.possible(train)
                if policy == "q_learning_fixed":
                    valid = [i for i in valid if ACTIONS[i][:2] == (3, 8)]
                action = learner.choose(state, valid, epsilon=max(.1, .8 - episode * .12))
                result = environment.act(train, val, test, action, regime=regime)
                # Development reward only; NO test metric enters Q-learning.
                reward = result["val_silhouette"] - cost_weight * (
                    result["total_seconds"] if cost_mode == "wall_time"
                    else (result["count_ae_full"] + 0.25 * result["count_ae_updates"])
                )
                if t + 1 < development_len:
                    next_train, _, _ = split_three(sequence[t + 1][1], seed + episode * 1000 + t + 1)
                    next_state = environment.state(next_train)
                else:
                    next_state = state
                next_valid = environment.possible(next_train) if t + 1 < development_len else environment.possible(train)
                if policy == "q_learning_fixed":
                    next_valid = [i for i in next_valid if ACTIONS[i][:2] == (3, 8)]
                learner.learn(state, action, reward, next_state, next_valid)
    development_seconds = time.perf_counter() - development_start
    warmup_start = time.perf_counter()
    # Evaluation replays the initial development data only to create memory;
    # Q weights remain frozen. This is independent of evaluation observations.
    environment = PolicyEnvironment(
        ae_epochs, update_epochs, seed=seed,
        reuse_threshold=0.65 if policy == "q_learning_gated" else None
    )
    for t, (_, x, regime) in enumerate(sequence[:development_len]):
        tr, val, test = split_three(x, seed + 99999 + t)
        state = environment.state(tr)
        valid = environment.possible(tr)
        if policy == "q_learning_fixed":
            valid = [i for i in valid if ACTIONS[i][:2] == (3, 8)]
        action = (learner.choose(state, valid, epsilon=0)
                  if policy in ("q_learning", "q_learning_gated", "q_learning_fixed")
                  else fixed_action(valid, policy))
        environment.act(tr, val, test, action, regime=regime)
    warmup_seconds = time.perf_counter() - warmup_start
    records = []
    before = environment.counters.copy()
    for t, (month, x, regime) in enumerate(sequence[development_len:]):
        tr, val, test = split_three(x, seed + 99999 + development_len + t)
        decision_start = time.perf_counter()
        state = environment.state(tr)
        valid = environment.possible(tr)
        if policy == "q_learning_fixed":
            valid = [i for i in valid if ACTIONS[i][:2] == (3, 8)]
        if policy in ("q_learning", "q_learning_gated", "q_learning_fixed"):
            action, decision_info = learner.choose_with_info(state, valid, epsilon=0)
        else:
            action = fixed_action(valid, policy)
            decision_info = dict(q_state=str(state), q_unseen_state=False,
                                 q_tie_count=0, q_tie_break=False, q_exploration=False,
                                 q_selected_value=float("nan"), q_best_value=float("nan"),
                                 q_margin=float("nan"), q_valid_values="{}")
        decision_seconds = time.perf_counter() - decision_start
        result = environment.act(tr, val, test, action, regime=regime)
        records.append(dict(batch=month, true_regime_for_audit=regime,
            method=policy, seed=seed, state=str(state), action_index=action,
            policy_training_updates=learner.updates,
            policy_training_seconds=development_seconds,
            memory_warmup_seconds=warmup_seconds,
            routing_seconds=decision_seconds,
            end_to_end_decision_seconds=result["total_seconds"] + decision_seconds,
            **decision_info, **result))
    totals = {field: environment.counters[field] - before[field] for field in before}
    return records, totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/aba_quality_cost.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--ae-epochs", type=int, default=2)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--cost-weight", type=float, default=.02)
    parser.add_argument("--cost-mode", choices=("wall_time", "operation_counts"), default="wall_time")
    parser.add_argument("--size", type=int, default=150)
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        # Development A-B-A; independent regenerated held-out sequence A-B-A
        # for the pilot. This is a replay of controlled regimes, NOT natural drift.
        seq = make_aba(seed=seed + 100, batches=12, size=args.size,
                       order=["A","A","B","B","A","A","A","A","B","B","A","A"])
        for policy in ("always_retrain", "reuse_when_available",
                       "always_update", "q_learning", "q_learning_gated"):
            records, totals = run_sequence(
                seq, seed=seed, policy=policy, development_episodes=args.episodes,
                development_len=6, ae_epochs=args.ae_epochs,
                update_epochs=args.update_epochs, cost_weight=args.cost_weight,
                cost_mode=args.cost_mode)
            rows.extend(records)
            print(json.dumps(dict(seed=seed, policy=policy, episodes=args.episodes,
                eval_batches=len(records),
                mean_test_silhouette=float(np.mean([r["test_silhouette"] for r in records])),
                evaluation_counters=totals)), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Output:", output)


if __name__ == "__main__":
    main()
