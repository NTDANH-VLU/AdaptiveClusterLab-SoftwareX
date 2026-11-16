"""FURAKI reduced DAMADICS external benchmark; no class-label leakage.

The FURAKI subset has 3,600 records, 32 features and a trailing 'drifting'
fault/category label. It is NOT the original full 2,246,400-record time series.
Block order should not be presented as original physical timestamps.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score,adjusted_rand_score

def assign(x,c):
    return np.argmin(((x[:,None,:]-c[None,:,:])**2).sum(axis=2),axis=1)

def metrics(x,lab):
    if not 1 < len(np.unique(lab)) < len(x):return np.nan,np.nan
    return float(silhouette_score(x,lab)),float(davies_bouldin_score(x,lab))

def prepare(path,block_size=450,seed=42):
    df=pd.read_csv(path)
    if "drifting" not in df.columns: raise ValueError("Expected FURAKI DAMADICS 'drifting' label column")
    y=df["drifting"].to_numpy()
    x=df.drop(columns=["drifting"]).apply(pd.to_numeric,errors="raise").to_numpy(dtype=np.float64)
    if x.shape[1]!=32 or not np.isfinite(x).all(): raise ValueError("Expected finite 32 numeric features")
    if len(x)%block_size: raise ValueError("Dataset not divisible by block size; retain full blocks only")
    # Fit normalization only on first block TRAIN subset, not on evaluation samples or later blocks.
    train0,test0=split_indices(block_size,seed)
    scaler=StandardScaler().fit(x[train0])
    transformed=scaler.transform(x)
    return [(str(i).zfill(2),transformed[i*block_size:(i+1)*block_size],
             y[i*block_size:(i+1)*block_size])
            for i in range(len(x)//block_size)]

def split_indices(count,seed):
    # Same stable row offsets used in every block, but each block has different records.
    h=np.array([int.from_bytes(hashlib.sha256(f"{seed}:{i}".encode()).digest()[:8],"big")/2**64
                for i in range(count)])
    tr=np.flatnonzero(h>=.3);te=np.flatnonzero(h<.3)
    return tr,te

def run(batches,method,seed,k=3,develop_blocks=4):
    frozen=last=None
    rows=[]
    for index,(block,x,y) in enumerate(batches):
        tr,te=split_indices(len(x),seed)
        start=time.perf_counter()
        fits=0;accepted=0
        if frozen is None:
            centers=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(x[tr]).cluster_centers_
            frozen=centers.copy();last=centers.copy();fits=1;decision="initial_fit"
        elif method=="frozen": centers=frozen.copy();decision="frozen"
        elif method=="always_reuse":centers=last.copy();accepted=1;decision="reuse"
        else:
            fresh=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(x[tr]).cluster_centers_
            fits=1
            if method=="memory_gate":
                old_sil,_=metrics(x[tr],assign(x[tr],last))
                new_sil,_=metrics(x[tr],assign(x[tr],fresh))
                accepted=int(np.isfinite(old_sil) and old_sil>new_sil)
                centers=last.copy() if accepted else fresh
                decision="reuse" if accepted else "refit"
            else:centers=fresh;decision="refit"
        compute=time.perf_counter()-start
        out=assign(x[te],centers)
        sil,dbi=metrics(x[te],out)
        ari=float(adjusted_rand_score(y[te],out))
        if method in ("refit","memory_gate"):last=centers.copy()
        rows.append(dict(block=block,phase="evaluation" if index>=develop_blocks else "development",
            method=method,seed=seed,k=k,n_train=len(tr),n_test=len(te),
            silhouette=sil,dbi=dbi,fault_label_ari=ari,fit_and_gate_seconds=compute,
            fit_calls=fits,memory_accepted=accepted,decision=decision))
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",default="results/damadics_baselines.csv")
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    p.add_argument("--ks",type=int,nargs="+",default=[3,5])
    a=p.parse_args()
    if not Path(a.input).is_file():raise FileNotFoundError(a.input)
    rows=[]
    for seed in a.seeds:
        batches=prepare(a.input,seed=seed)
        for k in a.ks:
            for method in ("frozen","refit","memory_gate","always_reuse"):
                rows.extend(run(batches,method,seed,k))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    for method in ("frozen","refit","memory_gate","always_reuse"):
        sample=[r for r in rows if r["method"]==method and r["phase"]=="evaluation"]
        print(json.dumps({"method":method,"evaluations":len(sample),
            "silhouette":float(np.nanmean([r["silhouette"] for r in sample])),
            "dbi":float(np.nanmean([r["dbi"] for r in sample])),
            "fault_ari":float(np.nanmean([r["fault_label_ari"] for r in sample])),
            "fits":sum(r["fit_calls"] for r in sample),
            "accepted":sum(r["memory_accepted"] for r in sample)}))
    print("wrote",out)

if __name__=="__main__":main()
