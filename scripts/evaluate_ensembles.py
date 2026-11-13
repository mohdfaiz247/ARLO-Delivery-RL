"""Predeclared mean-Q ensembles, always reported alongside individual runs."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
import pandas as pd
from delivery_rl.agents import load_agent,mean_q_ensemble
from delivery_rl.evaluation import evaluate,default_policies

def main(register=False):
    folder=ROOT/"results"/"improved_training"
    path=folder/"ensemble_protocol.json"
    if register:
        if path.exists(): raise RuntimeError("Protocol already registered")
        path.write_text(json.dumps({"status":"registered_before_final_results", "policy":"Arithmetic mean of all five learned Q tables per algorithm, not a best-seed selection",
            "training_seeds":[11,22,33,44,55],"validation_seed":82026,"validation_orders":2000,
            "final_seed":192026,"final_orders":5000,"reporting":"Report both ensembles and both individual-policy means, alongside unchanged baselines regardless of rankings",
            "selection":"Use the original two-seed validation-selected setting for each algorithm. Do not select ensemble members by test results."},indent=2))
        print("Registered ensemble protocol before final-test results.")
        return
    protocol=json.loads(path.read_text())
    meta=json.loads((folder/"metadata.json").read_text())
    if meta["status"]!="complete": raise RuntimeError("Wait for the individual-policy study to complete")
    ensembles={}
    for algorithm,setting in meta["selected_settings"].items():
        members=[load_agent(ROOT/"models"/"improved_training"/f"{algorithm}_{setting}_seed{seed}.npz") for seed in protocol["training_seeds"]]
        ensemble=mean_q_ensemble(members)
        ensemble.save(ROOT/"models"/"improved_training"/f"{algorithm}_ensemble.npz")
        ensembles[f"{algorithm} ensemble"]=ensemble
    for phase,seed,n in (("validation",protocol["validation_seed"],protocol["validation_orders"]),
                         ("final",protocol["final_seed"],protocol["final_orders"])):
        episodes,summary=evaluate(default_policies(ensembles),n,seed)
        wide=episodes.pivot(index="order_id",columns="policy",values="total_cost")
        rule=summary.loc[summary.policy=="Rule based","average_cost"].iloc[0]
        summary["gap_to_rule"]=summary.average_cost-rule
        summary["paired_rule_gap_ci95"]=[1.96*(wide[p]-wide["Rule based"]).std()/n**.5 for p in summary.policy]
        episodes.to_csv(folder/f"ensemble_{phase}_episodes.csv",index=False)
        summary.to_csv(folder/f"ensemble_{phase}_summary.csv",index=False)
        print(phase,summary[["policy","average_cost","gap_to_rule","paired_rule_gap_ci95"]].to_string(index=False),flush=True)
    protocol["status"]="complete"
    path.write_text(json.dumps(protocol,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--register",action="store_true")
    args=parser.parse_args(); main(args.register)
