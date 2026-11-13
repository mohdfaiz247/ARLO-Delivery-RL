import numpy as np
import pandas as pd
import pytest
from delivery_rl.agents import QLearningAgent,TrainingConfig
from delivery_rl.environment import N_STATES,EnvConfig
from delivery_rl.training import train
from delivery_rl.evaluation import evaluate,default_policies
from delivery_rl.studies import (supported_states,coverage,aggregate_seeds,choose_settings,
    economic_cases,run_studies,StudySetting)

def test_support_and_coverage_account_for_default_distribution():
    agent=QLearningAgent(TrainingConfig(episodes=300)); train(agent)
    support=supported_states()
    assert len(set(support))==len(support) and 0<len(support)<N_STATES
    assert set(np.flatnonzero(agent.visits.sum(axis=1))).issubset(set(support))
    result=coverage(agent)
    assert result["total_updates"]==900
    assert 0<result["support_state_coverage"]<=1
    assert 0<result["support_action_coverage"]<=1
    assert 0<=result["support_pairs_under_10_visits"]<=1

def test_seed_intervals_do_not_treat_orders_as_training_replicates():
    frame=pd.DataFrame({"policy":["q"]*5,"train_seed":[1,2,3,4,5],"average_cost":[10,12,14,16,18]})
    summary=aggregate_seeds(frame,["policy"]).iloc[0]
    assert summary.average_cost==14 and summary.seed_count==5
    assert summary.cost_seed_ci95==pytest.approx(2.776*np.std([10,12,14,16,18],ddof=1)/np.sqrt(5))
    with pytest.raises(ValueError): aggregate_seeds(pd.concat([frame,frame]),["policy"])

def test_selection_uses_only_validation_means_not_favoured_seed():
    validation=pd.DataFrame({"policy":["q"]*4,"setting":["a","a","b","b"],
        "train_seed":[1,2,1,2],"average_cost":[5,25,12,14]})
    assert choose_settings(validation)=={"q":"b"}

def test_unseen_decisions_and_budget_sweep():
    agent=QLearningAgent()
    rows,summary=evaluate(default_policies({"q":agent}),20,99,order_budget=0)
    q=summary[summary.policy=="q"].iloc[0]
    assert q.unseen_state_fraction==1 and q.unseen_action_fraction==1
    assert rows.initial_budget.eq(0).all() and rows.intervention_cost.eq(0).all()
    assert rows.budget_violations.eq(0).all()
    assert len(economic_cases())==12
    with pytest.raises(ValueError): evaluate(default_policies({}),20,99,order_budget=51)

def test_frozen_penalty_sweep_changes_cost_not_trajectory():
    agent=QLearningAgent(TrainingConfig(episodes=100)); train(agent)
    low,_=evaluate(default_policies({"q":agent}),30,18,EnvConfig(cost_per_late_day=7.5))
    high,_=evaluate(default_policies({"q":agent}),30,18,EnvConfig(cost_per_late_day=30.))
    assert low.late_days.equals(high.late_days)
    assert low.intervention_cost.equals(high.intervention_cost)
    assert np.allclose(high.total_cost-low.total_cost,22.5*low.late_days)

def test_study_pipeline_separates_outputs_and_records_protocol(tmp_path):
    import json
    result=run_studies(tmp_path,seeds=(1,2),settings=(StudySetting("tiny",10,.1),),
        evaluation_orders=4,progress=lambda message:None)
    assert set(result.policy)=={"q_learning","sarsa","No intervention","Rule based"}
    folder=tmp_path/"results"/"studies"
    meta=json.loads((folder/"metadata.json").read_text())
    assert meta["status"]=="complete"
    assert len({meta["validation_seed"],meta["final_test_seed"],meta["sensitivity_seed"]})==3
    assert (folder/"economic_sensitivity.html").exists()
    assert not (tmp_path/"models"/"q_learning.npz").exists()
    assert len(pd.read_csv(folder/"economics_runs.csv"))==2*12*4
