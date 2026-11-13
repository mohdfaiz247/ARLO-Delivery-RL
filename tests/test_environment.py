import numpy as np
import pytest
from delivery_rl.environment import DeliveryMitigationEnv,Order,EnvConfig,decode,risk,generate_orders,N_STATES

def test_three_steps_and_exact_accounting():
    env=DeliveryMitigationEnv(); state,info=env.reset(7,Order("T",4,2,1,40))
    assert 0 <= state < N_STATES and decode(state)==(0,7,1,2,40,0)
    reward=0
    for index,action in enumerate([1,2,0]):
        _,r,done,truncated,info=env.step(action); reward+=r
        assert env.stage==index+1 and not truncated
        assert done==(index==2)
    assert env.spent==33 and env.budget==7
    assert info["total_cost"]==-reward
    assert env.delay_cost==15*max(0,env.outlook)+(20 if env.outlook>0 else 0)
    assert env.get_state() is None and env.get_valid_actions()==[]
    with pytest.raises(ValueError): env.step(0)

@pytest.mark.parametrize("budget",range(51))
def test_budget_and_stage_masks(budget):
    env=DeliveryMitigationEnv(); env.reset(1,Order("T",2,0,0,budget))
    for _ in range(3):
        valid=env.get_valid_actions()
        assert 0 in valid
        assert all(env.config.costs[a]<=env.budget for a in valid)
        env.step(valid[-1]); assert env.budget>=0

def test_invalid_action_has_no_side_effects():
    env=DeliveryMitigationEnv(); env.reset(3,Order("T",2,0,0,0))
    before=env.info()
    with pytest.raises(ValueError): env.step(3)
    assert env.info()==before

def test_services_persist_without_double_charge():
    env=DeliveryMitigationEnv(); env.reset(1,Order("T",6,2,1,50))
    env.draws[:]=.5
    env.step(3)
    assert env.services==3 and env.spent==32 and env.budget==18
    assert env.get_valid_actions()==[0]
    env.step(0); assert env.get_valid_actions()==[0]
    env.step(0)
    assert env.spent==32 and env.history[-1]["saved_days"]>=1

@pytest.mark.parametrize("scenario",["pessimistic","moderate","optimistic"])
def test_stochastic_bounds(scenario):
    final=[]
    for seed in range(80):
        env=DeliveryMitigationEnv(EnvConfig(scenario)); env.reset(seed,Order("T",4,2,1,40))
        for _ in range(3):
            env.step(env.get_valid_actions()[-1]); info=env.info()
            assert -3<=env.outlook<=8 and 0<info["risk"]<1 and env.budget>=0
        final.append(env.outlook)
    assert len(set(final))>1

def test_matched_disruptions_independent_of_actions():
    envs=[DeliveryMitigationEnv(),DeliveryMitigationEnv()]
    for env in envs: env.reset(19,Order("T",3,2,1,50))
    for a in [3,0,0]: envs[0].step(a)
    for a in [0,0,0]: envs[1].step(a)
    assert [r["disturbance_days"] for r in envs[0].history]==[r["disturbance_days"] for r in envs[1].history]

def test_seed_and_order_coherence():
    assert generate_orders(20,4)==generate_orders(20,4)
    assert risk(6,2,1)>risk(0,2,1)>risk(-3,2,1)
    with pytest.raises(ValueError): Order("bad",99,0,0,40)

def test_no_action_can_stay_unchanged():
    env=DeliveryMitigationEnv(); env.reset(4,Order("T",0,0,0,40)); env.draws[:]=.5
    env.step(0); assert env.outlook==0
