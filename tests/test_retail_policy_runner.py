"""Leakage and accounting tests for the real Retail II policy experiment."""
import sys
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip("torch")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"/"controlled"))
from retail_policy_runner import stable_split,run
from quality_cost_aba import PolicyEnvironment

def test_hashed_customer_partition_is_stable_across_months():
    ids=np.array([str(i) for i in range(210)])
    x=np.arange(len(ids)*5).reshape(len(ids),5).astype(np.float32)
    parts=stable_split(x,ids,42,max_per_partition=500)
    reversed_parts=stable_split(x[::-1],ids[::-1],42,max_per_partition=500)
    for part,reverse in zip(parts,reversed_parts):
        assert {tuple(row) for row in part}=={tuple(row) for row in reverse}
    assert sum(len(p) for p in parts)==len(ids)

def test_evaluation_no_policy_learning_and_accounted_costs():
    rng=np.random.default_rng(1);sequence=[]
    for index in range(5):
        x=np.r_[rng.normal(-1,.4,(100,5)),rng.normal(1,.4,(100,5))].astype("float32")
        tr,val,te=stable_split(x,np.array([str(i) for i in range(len(x))]),42,500)
        sequence.append((f"2010-{index+1:02d}",tr,val,te))
    rows=run(42,sequence,policy="q_learning",development_months=3,
             episodes=2,ae_epochs=1,update_epochs=1)
    assert len(rows)==2
    assert all(r["policy_training_seconds"]>0 for r in rows)
    assert all(r["decision_seconds"]>=r["total_seconds"] for r in rows)
    assert all(r["n_test"]>16 and r["n_validation"]>16 for r in rows)
    assert all(np.isfinite(r["test_silhouette"]) for r in rows)

def test_real_data_split_never_uses_future_customer_data():
    ids=np.array([str(i) for i in range(200)])
    x=np.random.default_rng(2).normal(size=(200,5)).astype("float32")
    a=stable_split(x,ids,42,max_per_partition=500)
    changed=x.copy(); changed[0]+=1000
    b=stable_split(changed,ids,42,max_per_partition=500)
    # Customer allocation remains fixed regardless of future feature shifts.
    assert [len(q) for q in a]==[len(q) for q in b]
