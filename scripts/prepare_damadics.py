"""Generate reproducible FURAKI reduced DAMADICS blocks for policy research.

The 'drifting' column is excluded from all features and preprocessing. The
3,600-record reduced dataset is NOT the original full chronological DAMADICS
stream; the block index is a retained file order, not an industrial timestamp.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare(input_csv,output,block_size=450,development_blocks=4):
    data=pd.read_csv(input_csv)
    if "drifting" not in data.columns:raise ValueError("Expected FURAKI label 'drifting'")
    labels=data["drifting"].to_numpy()
    features=data.drop(columns=["drifting"]).apply(pd.to_numeric,errors="raise").to_numpy(dtype=np.float64)
    if features.ndim!=2 or features.shape[1]!=32 or not np.isfinite(features).all():
        raise ValueError("Expected 32 finite numeric input features")
    if len(features)%block_size:raise ValueError("Input does not divide into full blocks")
    blocks=len(features)//block_size
    if not 1<development_blocks<blocks:raise ValueError("Invalid developmental block count")
    scaler=StandardScaler().fit(features[:block_size*development_blocks])
    norm=scaler.transform(features)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for i in range(blocks):
        first=i*block_size;end=(i+1)*block_size
        filename=f"damadics_{i:02d}.npz"
        np.savez_compressed(out/filename,X=norm[first:end],
            customer_ids=np.array([f"sample_{j}" for j in range(first,end)],dtype=str),
            audit_labels=labels[first:end],
            feature_names=np.array(data.drop(columns=["drifting"]).columns,dtype=str))
        manifest.append({"block":i,"samples":block_size,"n_features":32,
            "phase":"development" if i<development_blocks else "evaluation",
            "file":filename})
    pd.DataFrame(manifest).to_csv(out/"manifest.csv",index=False)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",default="results/damadics")
    args=p.parse_args()
    print(prepare(args.input,args.output))
if __name__=="__main__":main()
