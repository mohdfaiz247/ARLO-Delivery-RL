"""Train, evaluate and export actual results. No dashboard required."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
import platform
from dataclasses import asdict
import numpy as np
import pandas as pd
from delivery_rl.agents import QLearningAgent,SarsaAgent,TrainingConfig
from delivery_rl.training import train
from delivery_rl.evaluation import evaluate,default_policies,robustness
from delivery_rl.charts import learning,comparison,scenario_chart,action_distribution

def run(episodes=15000,n=1000,seed=42,eval_seed=2026):
    result = ROOT/"results"; result.mkdir(exist_ok=True)
    agents = {}; curves = {}
    config = TrainingConfig(episodes=episodes,seed=seed)
    for cls in (QLearningAgent,SarsaAgent):
        agent = cls(config)
        curves[agent.algorithm] = train(agent)
        curves[agent.algorithm].to_csv(result/f"{agent.algorithm}_training.csv",index=False)
        agent.save(ROOT/"models"/f"{agent.algorithm}.npz")
        agents[agent.algorithm] = agent
    records,summary = evaluate(default_policies(agents),n,eval_seed)
    records.to_csv(result/"evaluation_episodes.csv",index=False)
    summary.to_csv(result/"evaluation_summary.csv",index=False)
    sensitivity = robustness(agents,n,eval_seed)
    sensitivity.to_csv(result/"robustness.csv",index=False)
    diagnostic,diagnostic_summary = evaluate(default_policies(agents),n,eval_seed+1,balanced=True)
    diagnostic.to_csv(result/"diagnostic_episodes.csv",index=False)
    diagnostic_summary.to_csv(result/"diagnostic_summary.csv",index=False)
    metadata = {"training":asdict(config),"evaluation_orders":n,"evaluation_seed":eval_seed,
        "training_scenario":"moderate","robustness_protocol":"Frozen moderate policies; matched orders and disturbances across scenarios",
        "diagnostic":"Uniform integer outlook -3..8; independent seed", "python":platform.python_version(),
        "numpy":np.__version__,"pandas":pd.__version__}
    (result/"experiment_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    figures = {"learning":learning(curves),"epsilon":learning(curves,"epsilon"),
               "cost":comparison(summary),"on_time":comparison(summary,"on_time_rate"),
               "actions":action_distribution(records),"robustness":scenario_chart(sensitivity)}
    for name,fig in figures.items():
        fig.write_html(result/f"{name}.html",include_plotlyjs=True)
    print(summary.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",type=int,default=15000)
    parser.add_argument("--evaluation-orders",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--eval-seed",type=int,default=2026)
    args = parser.parse_args()
    run(args.episodes,args.evaluation_orders,args.seed,args.eval_seed)
