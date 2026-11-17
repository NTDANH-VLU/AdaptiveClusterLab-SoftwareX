"""Prospective exploratory evaluation of the three-action quality/cost controller on Retail II.
NOT a reproduction of the author's published Hybrid-RL or CM-HRL models.
Stable customer-hash split; first 12 months for policy development, last 12 for test.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from quality_cost_aba import ACTIONS, PolicyEnvironment,TabularQLearning,fixed_action
POLICIES=("always_retrain","reuse_when_available","always_update","q_learning","q_learning_gated","q_learning_fixed")

def candidates(env, x, policy):
    valid = env.possible(x)
    return [i for i in valid if ACTIONS[i][:2] == (3,8)] if policy == "q_learning_fixed" else valid


def stable_split(x,ids,seed,max_per_partition=160):
    """Persist customer membership across months; train/validation/test 60/20/20."""
    h=np.array([int.from_bytes(hashlib.sha256(f"{seed}:{v}".encode()).digest()[:8],"big")/2**64 for v in ids])
    ranges=(np.where(h<.6)[0],np.where((h>=.6)&(h<.8))[0],np.where(h>=.8)[0])
    rng=np.random.default_rng(seed);result=[]
    for idx in ranges:
        if len(idx)>max_per_partition:
            idx=np.sort(rng.choice(idx,size=max_per_partition,replace=False))
        if len(idx)<16:raise ValueError("Too few distinct customers in split")
        result.append(x[idx])
    return result

def load(directory,seed,months=24,max_per_partition=160,prefix="retail"):
    ps=sorted(Path(directory).glob(f"{prefix}_*.npz"))
    if len(ps)<months:raise ValueError(f"Expected {months} months; found {len(ps)}")
    rows=[]
    for p in ps[:months]:
        with np.load(p,allow_pickle=False) as o:
            x=o["X"].astype(np.float32);ids=o["customer_ids"].astype(str)
        if x.ndim!=2 or x.shape[1]<2 or len(x)!=len(ids) or len(set(ids))!=len(ids):
            raise ValueError(f"Invalid monthly batch: {p}")
        tr,val,te=stable_split(x,ids,seed,max_per_partition)
        rows.append((p.stem.removeprefix(prefix+"_"),tr,val,te))
    return rows

def run(seed,sequence,policy,development_months=12,episodes=5,
        ae_epochs=2,update_epochs=1,cost_weight=.02,cost_mode="wall_time"):
    if cost_mode not in ("wall_time","operation_counts"):
        raise ValueError("unsupported cost mode")
    learner=TabularQLearning(seed)
    gate=.65 if policy=="q_learning_gated" else None
    start=time.perf_counter()
    if policy.startswith("q_learning"):
        for episode in range(episodes):
            env=PolicyEnvironment(ae_epochs,update_epochs,seed=seed,reuse_threshold=gate)
            for t,(_,tr,val,te) in enumerate(sequence[:development_months]):
                state=env.state(tr);valid=candidates(env,tr,policy)
                action=learner.choose(state,valid,epsilon=max(.1,.8-.12*episode))
                result=env.act(tr,val,te,action)
                reward=result["val_silhouette"]-cost_weight*(result["total_seconds"] if cost_mode=="wall_time" else result["count_ae_full"]+0.25*result["count_ae_updates"])
                if t+1<development_months:
                    next_tr=sequence[t+1][1];next_state=env.state(next_tr)
                    next_valid=candidates(env,next_tr,policy)
                else:next_state=state;next_valid=candidates(env,tr,policy)
                learner.learn(state,action,reward,next_state,next_valid)
    training_seconds=time.perf_counter()-start
    env=PolicyEnvironment(ae_epochs,update_epochs,seed=seed,reuse_threshold=gate)
    start=time.perf_counter()
    for _,tr,val,te in sequence[:development_months]:
        state=env.state(tr);valid=candidates(env,tr,policy)
        action=learner.choose(state,valid) if policy.startswith("q_learning") else fixed_action(valid,policy)
        env.act(tr,val,te,action)
    warmup_seconds=time.perf_counter()-start
    rows=[]
    for month,tr,val,te in sequence[development_months:]:
        start=time.perf_counter()
        state=env.state(tr);valid=candidates(env,tr,policy)
        if policy.startswith("q_learning"):
            action,diag=learner.choose_with_info(state,valid,epsilon=0)
        else:
            action=fixed_action(valid,policy)
            diag=dict(q_state=str(state),q_unseen_state=False,q_tie_count=0,
                      q_tie_break=False,q_exploration=False,
                      q_selected_value=float("nan"),q_best_value=float("nan"),
                      q_margin=float("nan"),q_valid_values="{}")
        routing=time.perf_counter()-start
        result=env.act(tr,val,te,action)
        rows.append(dict(dataset="temporally_batched_data",seed=seed,month=month,method=policy,
                         reward_cost_mode=cost_mode,
                         n_train=len(tr),n_validation=len(val),n_test=len(te),
                         policy_training_seconds=training_seconds,memory_warmup_seconds=warmup_seconds,
                         routing_seconds=routing,decision_seconds=routing+result["total_seconds"],**diag,**result))
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch-dir",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    p.add_argument("--months",type=int,default=24)
    p.add_argument("--prefix",default="retail",help="Monthly batch filename prefix")
    p.add_argument("--development-months",type=int,default=12)
    p.add_argument("--episodes",type=int,default=5)
    p.add_argument("--ae-epochs",type=int,default=2)
    p.add_argument("--update-epochs",type=int,default=1)
    p.add_argument("--cost-mode",choices=("wall_time","operation_counts"),default="wall_time")
    p.add_argument("--max-per-partition",type=int,default=160,
                   help="For bounded pilot compute; report subsampling explicitly")
    a=p.parse_args()
    rows=[]
    for seed in a.seeds:
        seq=load(a.batch_dir,seed,a.months,a.max_per_partition,a.prefix)
        for policy in POLICIES:
            part=run(seed,seq,policy,a.development_months,a.episodes,a.ae_epochs,a.update_epochs,cost_mode=a.cost_mode)
            rows.extend(part)
            print(json.dumps(dict(seed=seed,method=policy,n=len(part),
                mean_penalized_silhouette=float(np.mean([r["test_silhouette"] for r in part])),
                mean_valid_silhouette=float(np.nanmean([r["test_silhouette_valid"] for r in part])) if any(not r["test_degenerate"] for r in part) else None,
                n_degenerate=sum(r["test_degenerate"] for r in part),
                q_unseen=sum(r["q_unseen_state"] for r in part),
                q_tie_breaks=sum(r["q_tie_break"] for r in part),
                reuse=sum(r["operation"]=="reuse" for r in part),
                retrain=sum(r["count_ae_full"] for r in part))),flush=True)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(out)
if __name__=="__main__":main()
