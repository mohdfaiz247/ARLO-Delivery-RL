"""Verify a completed training improvement study and publish honest results."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
import pandas as pd
from delivery_rl.agents import load_agent
from delivery_rl.evaluation import evaluate,default_policies
from delivery_rl.charts import comparison,style
from delivery_rl.environment import EnvConfig
from delivery_rl.studies import aggregate_seeds
import plotly.express as px

def main():
    folder=ROOT/"results"/"improved_training"
    meta=json.loads((folder/"metadata.json").read_text())
    if meta["status"]!="complete": raise RuntimeError("Training study has not completed")
    final=pd.read_csv(folder/"final_summary.csv")
    runs=pd.read_csv(folder/"final_runs.csv")
    validation=pd.read_csv(folder/"all_validation_runs.csv")
    records=pd.concat([pd.read_csv(folder/f"final_episodes_seed{seed}.csv") for seed in meta["confirmation_seeds"]])
    assert len(records)==meta["final_orders"]*4*len(meta["confirmation_seeds"])
    assert records.budget_violations.sum()==0
    assert np.allclose(records.reward,-records.total_cost)
    assert np.allclose(records.total_cost,records.intervention_cost+records.delay_cost)
    for _,row in final.iterrows():
        assert np.isclose(row.average_cost,runs[runs.policy==row.policy].average_cost.mean())
    for algorithm,setting in meta["selected_settings"].items():
        for seed in meta["confirmation_seeds"]:
            agent=load_agent(ROOT/"models"/"improved_training"/f"{algorithm}_{setting}_seed{seed}.npz")
            assert agent.visits.sum()==3*agent.config.episodes
    # Compare old and new policies on the same final orders, solely as a
    # diagnostic after all training settings have been frozen.
    paired=[]; sensitivity=[]
    for seed in meta["confirmation_seeds"]:
        old_agents={}
        for alg in ("q_learning","sarsa"):
            old_path=ROOT/"models"/"studies"/f"{alg}_longer_60k_seed{seed}.npz"
            old_agents[alg]=load_agent(old_path)
        old_records,old_summary=evaluate(default_policies(old_agents),meta["final_orders"],meta["final_seed"])
        for alg in old_agents:
            old=old_records[old_records.policy==alg].set_index("order_id").total_cost
            new=records[(records.policy==alg)&(records.train_seed==seed)].set_index("order_id").total_cost
            diff=old-new
            paired.append(dict(policy=alg,train_seed=seed,old_cost=old.mean(),new_cost=new.mean(),
                saving_from_improved_training=diff.mean(),paired_order_ci95=1.96*diff.std()/np.sqrt(len(diff))))
        new_agents={a:load_agent(ROOT/"models"/"improved_training"/f"{a}_{setting}_seed{seed}.npz")
            for a,setting in meta["selected_settings"].items()}
        for penalty in (7.5,15.,30.):
            _,penalty_summary=evaluate(default_policies(new_agents),1000,212026,EnvConfig(cost_per_late_day=penalty))
            sensitivity.extend(penalty_summary.assign(train_seed=seed,late_day_penalty=penalty).to_dict("records"))
    paired=pd.DataFrame(paired); paired.to_csv(folder/"same_test_before_after.csv",index=False)
    sensitivity=pd.DataFrame(sensitivity)
    sensitivity.to_csv(folder/"penalty_sensitivity_runs.csv",index=False)
    aggregate_seeds(sensitivity,["policy","late_day_penalty"]).to_csv(folder/"penalty_sensitivity_summary.csv",index=False)
    fig=style(px.bar(final,x="policy",y="average_cost",color="policy",error_y="cost_seed_ci95",
        title="Extended training: fresh 5,000-order final test",
        labels={"average_cost":"Mean economic cost (R$/order)","policy":"Policy"}))
    fig.write_html(folder/"final_cost.html",include_plotlyjs=True)
    parts=["# Extended RL training: executed results","",
        "The simulator, costs, reward, order distribution and baseline rules are unchanged. "
        "The comparison explores 200,000-episode training, lower final exploration and an optional per-action visit-decay learning rate.","",
        f"Pilot: two independent seeds {meta['pilot_seeds']} per setting and algorithm, "
        f"evaluated on {meta['validation_orders']:,} validation orders at seed {meta['validation_seed']}. "
        f"Selection uses validation means only. Final evaluation freezes five policies per algorithm and uses "
        f"{meta['final_orders']:,} new matched orders at seed {meta['final_seed']}.","",
        "## Validation selection", ""]
    pilot=pd.read_csv(folder/"validation_runs.csv")
    for (algorithm,setting),part in pilot.groupby(["policy","setting"]):
        parts.append(f"- {algorithm}, {setting}: mean validation cost R${part.average_cost.mean():.4f}.")
    parts.extend(["",f"Selected configurations: {meta['selected_settings']}.","",
        "## Fresh final results",""])
    for _,row in final.iterrows():
        parts.append(f"- {row.policy}: mean total cost R${row.average_cost:.4f}, "
            f"on-time {row.on_time_rate:.2%}, savings versus no intervention R${row.savings:.4f}/order. "
            f"Approximate 95% interval across training-seed means: ±R${row.cost_seed_ci95:.4f}.")
    best=final.loc[final.average_cost.idxmin()]
    parts.extend(["",f"Lowest observed mean cost: **{best.policy}**, R${best.average_cost:.4f}/order.","",
        "The intervals above quantify training-seed variability conditional on the same orders. "
        "Non-learning baselines have zero training variation, but nonzero order-sampling uncertainty. "
        "Per-run paired cost-gap intervals versus the rule baseline are in final_runs.csv. "
        "Neither a small mean difference nor these seed intervals alone proves universal superiority.","",
        "## Before and after on identical orders",""])
    for alg,part in paired.groupby("policy"):
        parts.append(f"- {alg}: original 60k policies cost R${part.old_cost.mean():.4f}; "
            f"new selected 200k policies cost R${part.new_cost.mean():.4f}. "
            f"Mean saving from the training change: R${part.saving_from_improved_training.mean():.4f}/order.")
    parts.extend(["", "## Scope and limitations", "",
        "The optional step size is alpha_n(s,a) = alpha_0 / n(s,a)^p, with an exact per-pair visit count. "
        "The Q-learning and SARSA Bellman targets remain distinct and unchanged. "
        "Older saved models use p=0 by default and still load. "
        "The extended study also lowers final epsilon to 0.01, so any improvement compares a training package "
        "rather than isolating one causal hyperparameter. The two-seed pilot is a limited setting-selection sample. "
        "Real-world intervention effects remain synthetic and need calibration.","",
        "Earlier experiment results are preserved. Absolute costs from different test sets are not directly "
        "comparable. The before/after file evaluates old and new policies on identical fresh orders. "
        "No baseline has been weakened and no cost result has been replaced by an invented value.",""])
    (folder/"REPORT.md").write_text("\n".join(parts),encoding="utf-8")
    (folder/"verification.json").write_text(json.dumps({"status":"pass","final_episode_records":len(records),
        "budget_violations":int(records.budget_violations.sum()),"exact_cost_reward_accounting":True,
        "selected_model_updates_verified":True},indent=2))
    print("\n".join(parts),flush=True)

if __name__=="__main__": main()
