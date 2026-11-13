"""Frozen-policy evaluation on matched held-out orders and exogenous streams."""
import numpy as np
import pandas as pd
from dataclasses import replace
from .environment import DeliveryMitigationEnv, EnvConfig, generate_orders, risk

def choose(policy,env):
    valid = env.get_valid_actions()
    if policy == "no_intervention":
        return 0
    if policy == "rule_based":
        r = risk(env.outlook,env.order.distance,env.order.load)
        preferences = (3,2,1,0) if r >= .6 else ((1,0) if r >= .3 else (0,))
        return next(a for a in preferences if a in valid)
    return policy.select(env.get_state(),valid)

def evaluate(policies: dict, n: int = 1000, seed: int = 2026, config: EnvConfig | None = None, balanced=False, order_budget: int | None = None):
    if n < 2:
        raise ValueError("At least two evaluation orders required")
    orders = generate_orders(n,seed,balanced)
    if order_budget is not None:
        orders = [replace(order,initial_budget=order_budget) for order in orders]
    transition_seeds = np.random.default_rng(seed+19001).integers(0,2**31,size=n)
    rows = []
    for name,policy in policies.items():
        env = DeliveryMitigationEnv(config)
        for order,transition_seed in zip(orders,transition_seeds):
            env.reset(int(transition_seed),order); total_reward = 0; actions = []
            unseen_states = 0; unseen_actions = 0; incomplete_states = 0
            for _ in range(3):
                action = choose(policy,env); actions.append(action)
                if hasattr(policy,"visits"):
                    visits = policy.visits[env.get_state()]
                    unseen_states += int(visits.sum()==0)
                    unseen_actions += int(visits[action]==0)
                    incomplete_states += int(any(visits[a]==0 for a in env.get_valid_actions()))
                _,reward,done,_,info = env.step(action); total_reward += reward
                if done:
                    break
            rows.append({"policy":name,"order_id":order.order_id,"transition_seed":int(transition_seed),
                "initial_budget":order.initial_budget,"initial_outlook":order.outlook,
                "unseen_state_fraction":unseen_states/3 if hasattr(policy,"visits") else np.nan,
                "unseen_action_fraction":unseen_actions/3 if hasattr(policy,"visits") else np.nan,
                "incomplete_state_fraction":incomplete_states/3 if hasattr(policy,"visits") else np.nan,
                "total_cost":info["total_cost"],"intervention_cost":info["intervention_cost"],
                "delay_cost":info["delay_cost"],"late_days":info["late_days"],"on_time":int(info["late_days"]==0),
                "intervened":int(any(actions)),"reward":total_reward,"budget_violations":int(env.budget < 0),
                **{f"action_{a}":actions.count(a) for a in range(4)}})
    episodes = pd.DataFrame(rows)
    summary = episodes.groupby("policy",sort=False).agg(
        average_cost=("total_cost","mean"),cost_std=("total_cost","std"),
        on_time_rate=("on_time","mean"),average_late_days=("late_days","mean"),
        intervention_cost=("intervention_cost","mean"),intervention_frequency=("intervened","mean"),
        average_reward=("reward","mean"),budget_violations=("budget_violations","sum"))
    summary["cost_ci95"] = 1.96*summary.cost_std/np.sqrt(n)
    for metric in ("unseen_state_fraction","unseen_action_fraction","incomplete_state_fraction"):
        summary[metric] = episodes.groupby("policy")[metric].mean()
    for a in range(4):
        summary[f"action_{a}_fraction"] = episodes.groupby("policy")[f"action_{a}"].mean()/3
    baseline = summary.loc["No intervention","average_cost"] if "No intervention" in summary.index else np.nan
    summary["savings"] = baseline-summary.average_cost
    summary["savings_percent"] = 100*summary.savings/baseline if baseline > 0 else 0.
    # Paired cost difference intervals respect the matched-order protocol.
    pivot = episodes.pivot(index="order_id",columns="policy",values="total_cost")
    summary["savings_ci95"] = [1.96*(pivot["No intervention"]-pivot[p]).std()/np.sqrt(n)
        if "No intervention" in pivot else np.nan for p in summary.index]
    return episodes,summary.reset_index()

def default_policies(agents):
    return {**agents,"No intervention":"no_intervention","Rule based":"rule_based"}

def robustness(agents,n=1000,seed=2026):
    frames = []
    for scenario in ("pessimistic","moderate","optimistic"):
        _,summary = evaluate(default_policies(agents),n,seed,EnvConfig(scenario))
        frames.append(summary.assign(scenario=scenario))
    return pd.concat(frames,ignore_index=True)
