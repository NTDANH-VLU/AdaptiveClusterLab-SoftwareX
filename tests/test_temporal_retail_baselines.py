"""Simple invariants for fair temporal baseline methods."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from temporal_retail_baselines import split_customers, run_method, assign

def synthetic():
    r=np.random.default_rng(9)
    ids=np.array([str(i) for i in range(120)])
    return [
        ("2010-01",np.concatenate([r.normal(-1,0.4,(60,5)),r.normal(1,0.4,(60,5))]),ids),
        ("2010-02",np.concatenate([r.normal(-0.8,0.4,(60,5)),r.normal(1.2,0.4,(60,5))]),ids)
    ]

def test_customer_splits_are_stable():
    ids=np.array([str(i) for i in range(120)])
    tr,te=split_customers(ids,42)
    _,rev_te=split_customers(ids[::-1],42)
    assert set(ids[te])==set(ids[::-1][rev_te])
    assert not(set(ids[tr])&set(ids[te]))

def test_methods_and_fit_accounting():
    b=synthetic()
    for method in ("frozen","refit","always_reuse","memory_gate"):
        r=run_method(b,method,42,3,development_months=1)
        assert len(r)==2
        assert np.isfinite(r[1]["silhouette"])
        if method in ("frozen","always_reuse"):
            assert r[1]["kmeans_fit_calls"]==0
        else:
            assert r[1]["kmeans_fit_calls"]==1

def test_nearest_center_assignment():
    x=np.array([[0.1,0.1],[4.9,4.9]])
    assert assign(x,np.array([[0.,0.],[5.,5.]])).tolist()==[0,1]
