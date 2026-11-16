"""Transparent held-out baselines for temporal retail and electricity clustering.

This is a benchmark, NOT a reproduction of a published Hybrid-RL or CM-HRL policy.
All methods get identical batches, persistent customer-hash train/test splits,
the same K, and identical evaluation on held-out customer features.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

METHODS = ("frozen", "refit", "memory_gate", "always_reuse")

def load_batches(directory, count=24, prefix="retail", features=5):
    paths=sorted(Path(directory).glob(f"{prefix}_*.npz"))[:count]
    if len(paths)!=count: raise ValueError(f"Need {count} monthly files, found {len(paths)}")
    for path in paths:
        with np.load(path,allow_pickle=False) as payload:
            x=payload["X"].astype(np.float64)
            ids=payload["customer_ids"].astype(str)
        if x.ndim!=2 or x.shape[1]!=features or not np.isfinite(x).all():
            raise ValueError(f"Invalid batch: {path}")
        if len(ids)!=len(set(ids)): raise ValueError("Duplicate customer ID")
        yield path.stem.removeprefix(prefix+"_"),x,ids

def split_customers(ids,seed,fraction=.3):
    # A customer retains the same split assignment across every month.
    h=np.array([int.from_bytes(hashlib.sha256((str(seed)+":"+v).encode()).digest()[:8],"big")/2**64 for v in ids])
    train=np.flatnonzero(h>=fraction); test=np.flatnonzero(h<fraction)
    if min(len(train),len(test))<12: raise ValueError("Too few customers in split")
    return train,test

def assign(x,centers):
    return np.argmin(((x[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1)

def scores(x,labels):
    if not 1<len(np.unique(labels))<len(x): return float("nan"),float("nan")
    return float(silhouette_score(x,labels)),float(davies_bouldin_score(x,labels))

def run_method(batches, method, seed, k=3, development_months=12):
    if method not in METHODS: raise ValueError(method)
    frozen=last=None; prev={}; rows=[]
    for index,(month,x,ids) in enumerate(batches):
        tr,te=split_customers(ids,seed)
        xtr=x[tr]; xte=x[te]
        start=time.perf_counter()
        fit_calls=0; candidate=0; accepted=0
        if frozen is None:
            model=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(xtr)
            centers=model.cluster_centers_.copy()
            frozen=centers.copy();last=centers.copy()
            fit_calls=1;decision="initial_fit"
        elif method=="frozen":
            centers=frozen.copy();decision="frozen"
        elif method=="always_reuse":
            centers=last.copy();candidate=1;accepted=1;decision="reuse"
        elif method=="refit":
            model=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(xtr)
            centers=model.cluster_centers_.copy()
            fit_calls=1;decision="refit"
        else:
            model=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(xtr)
            fresh=model.cluster_centers_.copy()
            fit_calls=1;candidate=1
            # Select on TRAIN data only. Hold-out quality cannot affect decision.
            fresh_sil,_=scores(xtr,assign(xtr,fresh))
            old_sil,_=scores(xtr,assign(xtr,last))
            if np.isfinite(old_sil) and old_sil>fresh_sil:
                centers=last.copy();accepted=1;decision="reuse"
            else:
                centers=fresh;decision="refit"
        fit_seconds=time.perf_counter()-start
        sil,dbi=scores(xte,assign(xte,centers))
        labels=assign(x,centers)
        cur=dict(zip(ids,labels))
        overlap=sorted(prev.keys() & cur.keys())
        ari=(float(adjusted_rand_score([prev[i] for i in overlap],[cur[i] for i in overlap]))
             if len(overlap)>=max(12,3*k) else float("nan"))
        prev=cur
        if method in ("refit","memory_gate"): last=centers.copy()
        rows.append(dict(month=month,month_index=index,phase="evaluation" if index>=development_months else "development",
            method=method,k=k,seed=seed,n_train=len(tr),n_test=len(te),
            silhouette=sil,dbi=dbi,temporal_ari=ari,overlapping_customers=len(overlap),
            fitting_and_gate_seconds=fit_seconds,kmeans_fit_calls=fit_calls,
            memory_candidate=candidate,memory_accepted=accepted,decision=decision))
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch-dir",default="results/retail_ii")
    p.add_argument("--output",default="results/retail_ii/temporal_baselines.csv")
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--development-months",type=int,default=12)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    p.add_argument("--ks",type=int,nargs="+",default=[3,5])
    p.add_argument("--prefix", choices=("retail", "electricity"), default="retail")
    p.add_argument("--features", type=int, choices=(5,34), default=None)
    a=p.parse_args()
    expected= a.features if a.features is not None else (34 if a.prefix == "electricity" else 5)
    data=list(load_batches(a.batch_dir,a.months,prefix=a.prefix,features=expected))
    all_rows=[]
    for k in a.ks:
        for seed in a.seeds:
            for method in METHODS:
                all_rows.extend(run_method(data,method,seed,k,a.development_months))
                print("Completed",method,"K",k,"seed",seed,flush=True)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        w.writeheader();w.writerows(all_rows)
    for method in METHODS:
        sub=[r for r in all_rows if r["method"]==method and r["phase"]=="evaluation"]
        print(json.dumps(dict(method=method,n=len(sub),
            silhouette=float(np.nanmean([x["silhouette"] for x in sub])),
            dbi=float(np.nanmean([x["dbi"] for x in sub])),
            temporal_ari=float(np.nanmean([x["temporal_ari"] for x in sub])),
            fitting_and_gate_seconds=sum(x["fitting_and_gate_seconds"] for x in sub),
            kmeans_fit_calls=sum(x["kmeans_fit_calls"] for x in sub),
            memory_accepted=sum(x["memory_accepted"] for x in sub))),flush=True)
    print(path)

if __name__=="__main__": main()
