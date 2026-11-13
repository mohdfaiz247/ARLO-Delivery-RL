"""Validation-only RL tuning followed by one fresh five-seed final test.

Simulator, reward, order distribution and baseline rules are unchanged.
No final-test result is used to select a setting or a training seed.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import pandas as pd
from delivery_rl.agents import TrainingConfig,QLearningAgent,SarsaAgent,load_agent
from delivery_rl.training import train
from delivery_rl.evaluation import evaluate,default_policies
from delivery_rl.studies import coverage,aggregate_seeds

OUT=ROOT/"results"/"improved_training"
MODELS=ROOT/"models"/"improved_training"
# Predeclared grid. Existing validation orders may inform setting selection,
# but final seed 192026 has not appeared in the previous project studies.
SETTINGS={
    "constant_200k":dict(episodes=200000,learning_rate=.03,visit_decay=0.),
    "visit_decay_200k":dict(episodes=200000,learning_rate=1.,visit_decay=.6),
}
SEEDS=(11,22,33,44,55)
VALIDATION_SEED=82026
FINAL_SEED=192026

def fit(task):
    algorithm,setting,seed=task
    cls=QLearningAgent if algorithm=="q_learning" else SarsaAgent
    cfg=TrainingConfig(**SETTINGS[setting],seed=seed,epsilon_min=.01)
    agent=cls(cfg); history=train(agent)
    path=MODELS/f"{algorithm}_{setting}_seed{seed}.npz"
    agent.save(path)
    history.groupby((history.episode-1)//500).agg(episode=("episode","max"),
        block_mean_return=("return","mean"),epsilon=("epsilon","last")).to_csv(OUT/f"curve_{algorithm}_{setting}_seed{seed}.csv",index=False)
    _,summary=evaluate(default_policies({algorithm:agent}),2000,VALIDATION_SEED)
    row=summary[summary.policy==algorithm].iloc[0].to_dict()
    row.update(setting=setting,train_seed=seed,**coverage(agent))
    return row

def run(workers=2):
    OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
    metadata={"settings":SETTINGS,"pilot_seeds":[11,22],"confirmation_seeds":list(SEEDS),
        "validation_seed":VALIDATION_SEED,"validation_orders":2000,"final_seed":FINAL_SEED,
        "final_orders":5000,"epsilon_min":.01,"protocol":"Select each algorithm's setting using two-seed mean validation cost, then evaluate all five seeds once on fresh final orders",
        "unchanged":"Environment dynamics, moderate assumptions, cost reward, order distribution, rule baseline and no-action baseline",
        "status":"validation"}
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2))
    pilot=[]
    tasks=[(a,s,seed) for a in ("q_learning","sarsa") for s in SETTINGS for seed in (11,22)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(fit,tasks):
            pilot.append(row); pd.DataFrame(pilot).to_csv(OUT/"validation_runs.csv",index=False)
            print(f"Validation {row['policy']} {row['setting']} seed {row['train_seed']}: {row['average_cost']:.4f}",flush=True)
    pilot=pd.DataFrame(pilot)
    means=pilot.groupby(["policy","setting"]).average_cost.mean()
    chosen={a:means.loc[a].idxmin() for a in ("q_learning","sarsa")}
    metadata["selected_settings"]=chosen; metadata["status"]="confirmation_training"
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2))
    # Every confirmation seed is included. Retain the pilot models as trained.
    tasks=[(a,setting,seed) for a,setting in chosen.items() for seed in (33,44,55)]
    confirm=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(fit,tasks):
            confirm.append(row)
            print(f"Confirmation {row['policy']} seed {row['train_seed']}: trained",flush=True)
    pd.concat([pilot,pd.DataFrame(confirm)]).to_csv(OUT/"all_validation_runs.csv",index=False)
    metadata["status"]="fresh_final_test"
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2))
    final=[]
    for seed in SEEDS:
        agents={a:load_agent(MODELS/f"{a}_{setting}_seed{seed}.npz") for a,setting in chosen.items()}
        records,summary=evaluate(default_policies(agents),5000,FINAL_SEED)
        records.assign(train_seed=seed).to_csv(OUT/f"final_episodes_seed{seed}.csv",index=False)
        rule_cost=float(summary[summary.policy=="Rule based"].average_cost.iloc[0])
        # Paired order intervals vs rule are computed within each seed, separate
        # from across-training-seed variability.
        wide=records.pivot(index="order_id",columns="policy",values="total_cost")
        for row in summary.to_dict("records"):
            diffs=wide[row["policy"]]-wide["Rule based"]
            row.update(train_seed=seed,setting=chosen.get(row["policy"],"baseline"),
                gap_to_rule=row["average_cost"]-rule_cost,paired_rule_gap_ci95=1.96*diffs.std()/len(diffs)**.5)
            if row["policy"] in agents: row.update(coverage(agents[row["policy"]]))
            final.append(row)
        print(f"Final test completed for training seed {seed}",flush=True)
    final=pd.DataFrame(final); final.to_csv(OUT/"final_runs.csv",index=False)
    aggregate_seeds(final,["policy"]).to_csv(OUT/"final_summary.csv",index=False)
    metadata["status"]="complete"
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print(pd.read_csv(OUT/"final_summary.csv").to_string(index=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args(); run(args.workers)
