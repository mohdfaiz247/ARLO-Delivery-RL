import io
import numpy as np
import pytest
from delivery_rl.agents import QLearningAgent,SarsaAgent,TrainingConfig,load_agent

@pytest.mark.parametrize("cls",[QLearningAgent,SarsaAgent])
def test_visit_rate_is_local_and_follows_the_td_equation(cls):
    agent=cls(TrainingConfig(learning_rate=.5,visit_decay=.5))
    agent._apply(0,0,-10)
    assert agent.q[0,0]==-5
    agent._apply(0,0,-10)
    assert agent.q[0,0]==pytest.approx(-5+.5/np.sqrt(2)*(-5))
    agent._apply(0,1,-10)
    assert agent.q[0,1]==-5

def test_visit_rate_roundtrip_and_validation(tmp_path):
    agent=QLearningAgent(TrainingConfig(learning_rate=.5,visit_decay=.6))
    agent._apply(0,0,-10)
    path=tmp_path/"decay.npz"; agent.save(path)
    restored=load_agent(path)
    assert restored.config.visit_decay==.6
    restored._apply(0,0,-8); agent._apply(0,0,-8)
    assert np.array_equal(restored.q,agent.q)
    with pytest.raises(ValueError): TrainingConfig(visit_decay=-.1)

def test_legacy_model_metadata_without_decay_still_loads(tmp_path):
    import json
    agent=QLearningAgent()
    with np.load(io.BytesIO(agent.to_bytes()),allow_pickle=False) as data:
        meta=json.loads(str(data["metadata"]))
        del meta["training"]["visit_decay"]
        path=tmp_path/"legacy.npz"
        np.savez(path,q=data["q"],visits=data["visits"],metadata=json.dumps(meta))
    assert load_agent(path).config.visit_decay==0.

def test_ensemble_preserves_mean_values_and_explicit_provenance(tmp_path):
    from delivery_rl.agents import mean_q_ensemble
    a=QLearningAgent(TrainingConfig(seed=1)); b=QLearningAgent(TrainingConfig(seed=2))
    a.q[0]=[-4,-10,999,999]; b.q[0]=[-6,-8,999,999]
    a.visits[0,0]=2; b.visits[0,0]=3
    c=mean_q_ensemble([a,b])
    assert np.allclose(c.q[0],[-5,-9,999,999])
    assert c.visits[0,0]==5 and c.select(0,[0,1])==0
    path=tmp_path/"ensemble.npz"; c.save(path)
    loaded=load_agent(path)
    assert [m["seed"] for m in loaded.ensemble_members]==[1,2]
    with pytest.raises(ValueError): mean_q_ensemble([a,a])
