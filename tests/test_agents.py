import io
import json
import numpy as np
import pytest
from delivery_rl.agents import QLearningAgent,SarsaAgent,TrainingConfig,load_agent
from delivery_rl.environment import N_STATES,EnvConfig
from delivery_rl.training import train

def test_q_update_and_mask():
    a=QLearningAgent(TrainingConfig(learning_rate=.5,gamma=.9))
    assert a.q.shape==(N_STATES,4)
    a.q[1]=[2,10,999,999]
    a.update(0,0,-3,1,[0,1],False)
    assert a.q[0,0]==pytest.approx(3.)
    assert a.select(1,[0,1])==1
    a.update(0,1,-10,None,[],True)
    assert a.q[0,1]==-5

def test_sarsa_actually_selected_next_action():
    cfg=TrainingConfig(learning_rate=.5,gamma=.9)
    q=QLearningAgent(cfg); s=SarsaAgent(cfg)
    for a in [q,s]: a.q[1]=[2,10,999,999]
    q.update(0,0,-3,1,[0,1],False)
    s.update(0,0,-3,1,0,False)
    assert q.q[0,0]==3 and s.q[0,0]==pytest.approx(-.6)
    s.update(0,1,-10,None,None,True); assert s.q[0,1]==-5

@pytest.mark.parametrize("cls",[QLearningAgent,SarsaAgent])
def test_exploration_and_masking(cls):
    a=cls()
    selected=[a.select(0,[0,2],1.) for _ in range(100)]
    assert set(selected)=={0,2}
    assert a.select(0,[0,2],0.)==0
    with pytest.raises(ValueError): a.select(0,[])

@pytest.mark.parametrize("cls",[QLearningAgent,SarsaAgent])
def test_training_reproducible_and_learns(cls):
    cfg=TrainingConfig(episodes=120,seed=10)
    a,b=cls(cfg),cls(cfg)
    fa,fb=train(a),train(b)
    assert fa.equals(fb) and np.array_equal(a.q,b.q)
    assert np.count_nonzero(a.q)>0
    assert np.allclose(fa["return"],-fa.total_cost)
    assert a.visits.sum()==360

def test_save_load_configuration_validation(tmp_path):
    a=QLearningAgent(TrainingConfig(episodes=10)); train(a)
    path=tmp_path/"agent.npz"; a.save(path); b=load_agent(path,EnvConfig())
    assert np.array_equal(a.q,b.q) and np.array_equal(a.visits,b.visits)
    assert a.config==b.config
    with pytest.raises(ValueError): load_agent(path,EnvConfig("optimistic"))
    with np.load(io.BytesIO(a.to_bytes()),allow_pickle=False) as data:
        meta=json.loads(str(data["metadata"])); meta["encoding"]="wrong"
        np.savez(path,q=data["q"],visits=data["visits"],metadata=json.dumps(meta))
    with pytest.raises(ValueError): load_agent(path)

def test_epsilon_schedule_and_validation():
    c=TrainingConfig(episodes=100)
    values=[c.epsilon(i) for i in range(100)]
    assert values[0]==pytest.approx(1.) and values[-1]==pytest.approx(.05)
    assert all(x>=y for x,y in zip(values,values[1:]))
    with pytest.raises(ValueError): TrainingConfig(epsilon_start=.1,epsilon_min=.2)
