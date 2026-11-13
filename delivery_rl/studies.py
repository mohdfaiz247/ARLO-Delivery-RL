"""Repeated-seed studies with validation-only selection and a fresh final test set.

Results distinguish training-seed variability from order-sampling uncertainty.
No test outcome is used to select hyperparameters or a favoured training seed.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import plotly.express as px
from .agents import QLearningAgent, SarsaAgent, TrainingConfig, load_agent
from .environment import EnvConfig, STATE_DIMS, N_STATES, valid_actions
from .training import train
from .evaluation import evaluate, default_policies
from .charts import style

@dataclass(frozen=True)
class StudySetting:
    label: str
    episodes: int
    learning_rate: float

DEFAULT_SETTINGS = (
    StudySetting("reference_15k",15000,.1),
    StudySetting("longer_60k",60000,.1),
    StudySetting("longer_lower_alpha",60000,.03),
)
DEFAULT_SEEDS = (11,22,33,44,55)

def supported_states(config: EnvConfig | None = None) -> np.ndarray:
    """Support under the default generator for all three non-degenerate scenarios.

    Every generated initial integer outlook -3..7 has positive probability.
    At later decision stages all -3..8 outlooks have positive probability for
    every reachable budget/service pair: each service may fail, disruptions
    can increase outlook by two, and clipping preserves the lower endpoint.
    This is distribution-specific, not support under every manual input.
    """
    config = config or EnvConfig()
    pairs = {(b,0) for b in (0,8,25,32,40,50)}
    states = []
    for stage in range(3):
        next_pairs = set()
        for budget,services in sorted(pairs):
            for outlook in range(-3,8 if stage==0 else 9):
                for load in range(2):
                    for distance in range(3):
                        states.append(np.ravel_multi_index((stage,outlook+3,load,distance,budget,services),STATE_DIMS))
            for action in valid_actions(stage,budget,services,config.costs):
                bits = (1 if action in (1,3) else 0) | (2 if action in (2,3) else 0)
                next_pairs.add((budget-config.costs[action],services|bits))
        pairs = next_pairs
    return np.asarray(states,dtype=np.int64)

def coverage(agent) -> dict:
    support = supported_states(agent.env_config)
    counts = agent.visits.sum(axis=1)
    feasible_pairs = []
    for state in support:
        stage,_,_,_,budget,services = np.unravel_index(state,STATE_DIMS)
        feasible_pairs.extend((int(state),a) for a in valid_actions(stage,budget,services,agent.env_config.costs))
    pair_counts = np.asarray([agent.visits[s,a] for s,a in feasible_pairs])
    return {"allocated_states":N_STATES,"supported_states":len(support),
        "visited_states":int(np.count_nonzero(counts)),
        "support_state_coverage":float(np.mean(counts[support]>0)),
        "support_action_coverage":float(np.mean(pair_counts>0)),
        "support_pairs_under_10_visits":float(np.mean(pair_counts<10)),
        "median_support_pair_visits":float(np.median(pair_counts)),
        "total_updates":int(agent.visits.sum())}

def aggregate_seeds(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Student-t intervals across seed means, conditional on shared test orders.

    For 2..10 seeds use explicit two-sided .975 t critical values. Larger
    samples use a conservative critical value of 2.262 (the df=9 value).
    One seed cannot estimate training variability and returns NaN intervals.
    """
    critical = {2:12.706,3:4.303,4:3.182,5:2.776,6:2.571,7:2.447,8:2.365,9:2.306,10:2.262}
    rows = []
    for keys,part in frame.groupby(groups,sort=False):
        if not isinstance(keys,tuple): keys=(keys,)
        if part.train_seed.duplicated().any():
            raise ValueError("Each group must have one mean per independent training seed")
        n = len(part); std=float(part.average_cost.std(ddof=1))
        row = dict(zip(groups,keys))
        row.update(seed_count=n,average_cost=float(part.average_cost.mean()),cost_seed_std=std,
                   cost_seed_ci95=critical.get(n,2.262)*std/np.sqrt(n) if n>1 else np.nan)
        for name in ("on_time_rate","average_late_days","intervention_cost","savings","gap_to_rule",
                     "support_state_coverage","support_action_coverage","support_pairs_under_10_visits",
                     "unseen_state_fraction","unseen_action_fraction","incomplete_state_fraction"):
            if name in part: row[name]=float(part[name].mean())
        for name in ("savings","gap_to_rule"):
            if name in part:
                row[f"{name}_seed_ci95"] = critical.get(n,2.262)*float(part[name].std(ddof=1))/np.sqrt(n) if n>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def choose_settings(validation: pd.DataFrame) -> dict[str,str]:
    means = validation.groupby(["policy","setting"],sort=True).average_cost.mean()
    return {algorithm:str(means.loc[algorithm].idxmin()) for algorithm in means.index.get_level_values(0).unique()}

def economic_cases():
    cases = []
    for budget in (0,8,25,32,40,50):
        cases.append(("Budget",float(budget),EnvConfig(),budget))
    for multiplier in (.75,1.,1.25):
        costs = tuple(int(round(c*multiplier)) for c in EnvConfig().costs)
        cases.append(("Intervention price multiplier",multiplier,EnvConfig(costs=costs),None))
    for per_day in (7.5,15.,30.):
        cases.append(("Late-day penalty",per_day,EnvConfig(cost_per_late_day=per_day),None))
    return cases

def export_figures(folder):
    folder=Path(folder)
    validation=pd.read_csv(folder/"validation_summary.csv")
    final=pd.read_csv(folder/"final_summary.csv")
    cost=style(px.bar(final,x="policy",y="average_cost",color="policy",error_y="cost_seed_ci95",
        title="Fresh test set: mean economic cost across training seeds",
        labels={"average_cost":"Mean total cost (R$/order)","policy":"Policy"}))
    selection=style(px.bar(validation,x="setting",y="average_cost",color="policy",barmode="group",error_y="cost_seed_ci95",
        title="Validation study: training budget and learning rate",
        labels={"average_cost":"Mean total cost (R$/order)","setting":"Predefined training configuration","policy":"Algorithm"}))
    sensitivity=pd.read_csv(folder/"economics_summary.csv")
    sensitivity["factor_units"]=sensitivity.factor.map({"Budget":"Starting budget (R$)",
        "Intervention price multiplier":"Intervention price (× default)","Late-day penalty":"Late-day penalty (R$/day)"})
    sensitivity_fig=style(px.line(sensitivity,x="value",y="savings",color="policy",facet_col="factor_units",
        markers=True,title="Fixed-policy economic stress tests: savings versus no intervention",
        labels={"value":"Factor value (units in panel title)","savings":"Mean saving (R$/order)","factor_units":"Factor"}))
    sensitivity_fig.update_xaxes(matches=None)
    for name,fig in (("repeated_seed_cost",cost),("training_study",selection),("economic_sensitivity",sensitivity_fig)):
        fig.write_html(folder/f"{name}.html",include_plotlyjs=True)

def write_report(folder):
    """Generate readable findings from persisted, executed experiment outputs."""
    folder=Path(folder)
    meta=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    final=pd.read_csv(folder/"final_summary.csv")
    validation=pd.read_csv(folder/"validation_summary.csv")
    runs=pd.read_csv(folder/"validation_runs.csv")
    economics=pd.read_csv(folder/"economics_summary.csv")
    text=["# Repeated-seed research findings", "",
        f"Executed protocol: {len(meta['training_seeds'])} independent training seeds {meta['training_seeds']}; "
        f"{meta['evaluation_orders']:,} matched orders in each evaluation. Validation seed {meta['validation_seed']}; "
        f"fresh final test seed {meta['final_test_seed']}; economic sensitivity seed {meta['sensitivity_seed']}.","",
        "## Predefined training study and coverage", "",
        "Only validation means select the training setting. No final test result or favoured training seed is used for selection.",""]
    for setting in meta["settings"]:
        text.append(f"- {setting['label']}: {setting['episodes']:,} episodes, learning rate {setting['learning_rate']}.")
    text.extend(["", "Allocated table size and distribution-supported size are different:"])
    for algorithm in meta["selected_settings"]:
        ref=validation[validation.policy==algorithm]
        for _,row in ref.iterrows():
            actual=runs[(runs.policy==algorithm)&(runs.setting==row.setting)].iloc[0]
            text.append(f"- {algorithm}, {row.setting}: validation cost R${row.average_cost:.3f}; "
                f"{int(actual.supported_states):,} supported states out of {int(actual.allocated_states):,} allocated; "
                f"state coverage {row.support_state_coverage:.1%}, feasible-action coverage {row.support_action_coverage:.1%}; "
                f"{row.support_pairs_under_10_visits:.1%} of supported feasible pairs have fewer than 10 visits. "
                f"Unvisited chosen actions on validation: {row.unseen_action_fraction:.2%}.")
    text.extend(["", "The support count is specific to the default generator and configured costs. Coverage is not a guarantee of accurate values. "
        "Lossless reindexing could remove unreachable allocated rows, but would not pool samples or improve statistical efficiency. "
        "The exact budget/service representation is retained to avoid hiding feasible decisions.","",
        "## Fresh final test", ""])
    for _,row in final.iterrows():
        text.append(f"- {row.policy}: mean cost R${row.average_cost:.3f}; training-seed standard deviation R${row.cost_seed_std:.3f}; "
            f"approximate 95% seed-mean interval ±R${row.cost_seed_ci95:.3f}; on-time {row.on_time_rate:.1%}; "
            f"saving versus no intervention R${row.savings:.3f}/order.")
    best=final.loc[final.average_cost.idxmin()]
    text.extend(["",f"Lowest observed final-test mean: **{best.policy}**, R${best.average_cost:.3f}/order. "
        "This is a comparison of observed means, not a universal superiority claim.",""])
    for algorithm,setting in meta["selected_settings"].items():
        row=final[final.policy==algorithm].iloc[0]
        text.append(f"- {algorithm} selected {setting} on validation; final mean gap to the rule baseline "
            f"is R${row.gap_to_rule:+.3f}/order (positive means RL is more expensive).")
    text.extend(["", "Error bars use approximate Student-t intervals across training-seed means, conditional on the same final orders. "
        "The non-learning baselines are repeated on the identical order set and therefore have zero training-seed variability, "
        "not zero order uncertainty. Per-run cost and paired-savings Monte Carlo intervals are in final_runs.csv. "
        "These are separate uncertainty estimates; no combined real-world or hierarchical confidence claim is made.","",
        "## Economic sensitivity: frozen-policy stress tests", "",
        "Selected policies are not retrained for changed prices or penalties. Budgets are already represented in mixed-budget training. "
        "Price changes can change both action feasibility and remaining-budget support; unseen-action fractions help distinguish "
        "coverage shift from the change in economics.",""])
    for algorithm in meta["selected_settings"]:
        for factor in economics.factor.unique():
            part=economics[(economics.policy==algorithm)&(economics.factor==factor)]
            parts=[f"{r.value:g}: R${r.savings:+.3f} saved/order (unvisited chosen actions {r.unseen_action_fraction:.1%})" for _,r in part.iterrows()]
            text.append(f"- {algorithm}, {factor}: "+"; ".join(parts)+".")
    text.extend(["", "## Interpretation and limits", "",
        "Use the validation learning-rate and episode comparisons to assess whether additional experience changes performance. "
        "Do not conclude that allocated table size alone caused the original RL losses. Even a visited action can have a noisy value; "
        "constant learning rates, exploration, finite sampling and a useful rule baseline remain plausible explanations. "
        "This experiment does not isolate every causal contributor.","",
        "All parameters and orders are synthetic. Five training seeds are a stronger repeatability check, not proof of convergence. "
        "Reported settings are selected within a small predefined grid. New environment prices/rewards need retraining for optimal "
        "adaptation; frozen stress tests demonstrate robustness limits. Real-world deployment requires calibration and external validation.","",
        "Reproduce with `python scripts/run_studies.py`. Original single-seed outputs remain separate in results/ and models/. "
        "Research model snapshots live in models/studies/. See PROFESSOR_DEMO_GUIDE.md for the presentation sequence.",""])
    (folder/"REPORT.md").write_text("\n".join(text),encoding="utf-8")

def run_studies(root, seeds=DEFAULT_SEEDS, settings=DEFAULT_SETTINGS, evaluation_orders=1000,
                validation_seed=82026, test_seed=92026, sensitivity_seed=102026, progress=print):
    if len(seeds)<2 or len(set(seeds))!=len(seeds):
        raise ValueError("At least two distinct training seeds required")
    if len({validation_seed,test_seed,sensitivity_seed})!=3:
        raise ValueError("Validation, final test and sensitivity seeds must differ")
    if not settings or len({s.label for s in settings})!=len(settings):
        raise ValueError("Settings must be nonempty with distinct labels")
    root=Path(root); output=root/"results"/"studies"; output.mkdir(parents=True,exist_ok=True)
    model_dir=root/"models"/"studies"; model_dir.mkdir(parents=True,exist_ok=True)
    metadata={"training_seeds":list(seeds),"settings":[asdict(s) for s in settings],
        "evaluation_orders":evaluation_orders,"validation_seed":validation_seed,"final_test_seed":test_seed,
        "sensitivity_seed":sensitivity_seed,"selection":"Lowest mean validation cost per algorithm; never select by final test or best seed",
        "intervals":"Student-t across independent training seed means, conditional on shared orders; per-run MC intervals separate",
        "economics":"Frozen selected policies, changed budgets/prices/penalties; no retraining under changed reward or prices",
        "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"status":"running"}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    validation=[]
    for cls in (QLearningAgent,SarsaAgent):
        for setting in settings:
            for seed in seeds:
                agent=cls(TrainingConfig(episodes=setting.episodes,learning_rate=setting.learning_rate,seed=int(seed)))
                history=train(agent)
                agent.save(model_dir/f"{agent.algorithm}_{setting.label}_seed{seed}.npz")
                # Actual nonoverlapping 250-episode block means, not invented curves.
                blocks=history.groupby((history.episode-1)//250).agg(
                    episode=("episode","max"),block_mean_return=("return","mean"),
                    epsilon=("epsilon","last"),mean_interventions=("interventions","mean"))
                blocks.to_csv(output/f"curve_{agent.algorithm}_{setting.label}_seed{seed}.csv",index=False)
                _,summary=evaluate(default_policies({agent.algorithm:agent}),evaluation_orders,validation_seed)
                result=summary.loc[summary.policy==agent.algorithm].iloc[0].to_dict()
                result.update(setting=setting.label,train_seed=int(seed),episodes=setting.episodes,
                    learning_rate=setting.learning_rate,gap_to_rule=float(result["average_cost"]-summary.loc[summary.policy=="Rule based","average_cost"].iloc[0]),
                    **coverage(agent))
                validation.append(result)
                pd.DataFrame(validation).to_csv(output/"validation_runs.csv",index=False)
                progress(f"Validation {agent.algorithm} / {setting.label} / seed {seed}: R$ {result['average_cost']:.3f}")
    validation=pd.DataFrame(validation)
    aggregate_seeds(validation,["policy","setting"]).to_csv(output/"validation_summary.csv",index=False)
    selected=choose_settings(validation)
    metadata["selected_settings"]=selected
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    final=[]; economics=[]
    for seed in seeds:
        agents={name:load_agent(model_dir/f"{name}_{setting}_seed{seed}.npz") for name,setting in selected.items()}
        records,summary=evaluate(default_policies(agents),evaluation_orders,test_seed)
        records.assign(train_seed=seed).to_csv(output/f"final_episodes_seed{seed}.csv",index=False)
        rule_cost=float(summary.loc[summary.policy=="Rule based","average_cost"].iloc[0])
        for row in summary.to_dict("records"):
            row.update(train_seed=int(seed),gap_to_rule=row["average_cost"]-rule_cost,
                       setting=selected.get(row["policy"],"baseline"))
            if row["policy"] in agents: row.update(coverage(agents[row["policy"]]))
            final.append(row)
        for factor,value,config,budget in economic_cases():
            _,summary=evaluate(default_policies(agents),evaluation_orders,sensitivity_seed,config,order_budget=budget)
            summary=summary.assign(train_seed=seed,factor=factor,value=value)
            economics.extend(summary.to_dict("records"))
        progress(f"Fresh test and 12 economic cases completed for training seed {seed}")
    final=pd.DataFrame(final); final.to_csv(output/"final_runs.csv",index=False)
    aggregate_seeds(final,["policy"]).to_csv(output/"final_summary.csv",index=False)
    economics=pd.DataFrame(economics); economics.to_csv(output/"economics_runs.csv",index=False)
    aggregate_seeds(economics,["policy","factor","value"]).to_csv(output/"economics_summary.csv",index=False)
    metadata["status"]="complete"
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    export_figures(output)
    write_report(output)
    return pd.read_csv(output/"final_summary.csv")
