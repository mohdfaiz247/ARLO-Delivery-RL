import numpy as np
from delivery_rl.agents import QLearningAgent,TrainingConfig
from delivery_rl.evaluation import evaluate,default_policies,robustness
from delivery_rl.training import train

def test_evaluation_metrics_pairing_and_freeze():
    agent=QLearningAgent(TrainingConfig(episodes=50)); train(agent)
    original=agent.q.copy()
    rows,summary=evaluate(default_policies({"q_learning":agent}),80,902)
    rows2,summary2=evaluate(default_policies({"q_learning":agent}),80,902)
    assert rows.equals(rows2) and summary.equals(summary2)
    assert np.array_equal(original,agent.q)
    assert rows.groupby("order_id").transition_seed.nunique().eq(1).all()
    for _,record in summary.iterrows():
        episodes=rows[rows.policy==record.policy]
        assert np.isclose(episodes.total_cost.mean(),record.average_cost)
        assert np.isclose((episodes.late_days==0).mean(),record.on_time_rate)
        assert np.isclose(record.average_reward,-record.average_cost)
        assert record.budget_violations==0
    assert len(rows)==240

def test_baselines_and_robustness_without_models():
    records,summary=evaluate(default_policies({}),50,8)
    baseline=records[records.policy=="No intervention"]
    assert baseline.intervention_cost.eq(0).all() and baseline.action_0.eq(3).all()
    frame=robustness({},30,8)
    assert len(frame)==6 and set(frame.scenario)=={"pessimistic","moderate","optimistic"}
