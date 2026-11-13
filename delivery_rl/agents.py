"""Distinct NumPy temporal-difference agents; no policy thresholds in RL."""
from dataclasses import dataclass, asdict
from pathlib import Path
import io
import json
import numpy as np
from .environment import N_STATES, STATE_DIMS, ENCODING, EnvConfig

@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 15000
    learning_rate: float = .1
    gamma: float = 1.
    epsilon_start: float = 1.
    epsilon_min: float = .05
    decay_fraction: float = .85
    seed: int = 42
    visit_decay: float = 0.

    def __post_init__(self):
        if self.episodes < 1 or not 0 < self.learning_rate <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid training configuration")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1 or not 0 < self.decay_fraction <= 1:
            raise ValueError("Invalid exploration configuration")
        if not 0 <= self.visit_decay <= 1:
            raise ValueError("Visit decay must be between zero and one")

    def epsilon(self, episode):
        fraction = min(1.,episode/max(1,self.episodes*self.decay_fraction))
        # Smooth exponential interpolation; supports an exact zero endpoint.
        weight = (np.exp(-5*fraction)-np.exp(-5))/(1-np.exp(-5))
        return float(self.epsilon_min+(self.epsilon_start-self.epsilon_min)*weight)

class TabularAgent:
    algorithm = "base"
    def __init__(self, config: TrainingConfig | None = None, env_config: EnvConfig | None = None):
        self.config = config or TrainingConfig(); self.env_config = env_config or EnvConfig()
        self.q = np.zeros((N_STATES,4),dtype=np.float64)
        self.visits = np.zeros((N_STATES,4),dtype=np.int32)
        self.rng = np.random.default_rng(self.config.seed)
        self.ensemble_members = []

    def select(self, state: int, valid: list[int], epsilon: float = 0.) -> int:
        if not valid:
            raise ValueError("Cannot select from an empty action set")
        if epsilon and self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        # Stable ties make frozen evaluation independent of policy RNG state.
        return int(valid[int(np.argmax(self.q[state,valid]))])

    def _apply(self,state,action,target):
        # Constant rate remains the backwards-compatible default. An optional
        # per-pair power schedule reduces sampling noise in well-visited pairs.
        count = int(self.visits[state,action]) + 1
        alpha = self.config.learning_rate / count**self.config.visit_decay
        self.q[state,action] += alpha*(target-self.q[state,action])
        self.visits[state,action] += 1

    def to_bytes(self) -> bytes:
        metadata = {"algorithm": self.algorithm,"encoding": ENCODING,"dimensions": STATE_DIMS,
                    "training": asdict(self.config),"environment": asdict(self.env_config),
                    "ensemble_members":self.ensemble_members}
        buffer = io.BytesIO()
        np.savez_compressed(buffer,q=self.q,visits=self.visits,metadata=json.dumps(metadata))
        return buffer.getvalue()

    def save(self,path):
        path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(self.to_bytes())

class QLearningAgent(TabularAgent):
    algorithm = "q_learning"
    def update(self,state,action,reward,next_state,valid_next,terminated):
        target = reward if terminated else reward+self.config.gamma*np.max(self.q[next_state,valid_next])
        self._apply(state,action,target)

class SarsaAgent(TabularAgent):
    algorithm = "sarsa"
    def update(self,state,action,reward,next_state,next_action,terminated):
        target = reward if terminated else reward+self.config.gamma*self.q[next_state,next_action]
        self._apply(state,action,target)

def load_agent(path, expected_env: EnvConfig | None = None) -> TabularAgent:
    with np.load(path,allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta["encoding"] != ENCODING or tuple(meta["dimensions"]) != STATE_DIMS:
            raise ValueError("Incompatible state encoding")
        env_data = meta["environment"]; env_data["costs"] = tuple(env_data["costs"])
        env_config = EnvConfig(**env_data)
        if expected_env is not None and env_config != expected_env:
            raise ValueError("Saved environment does not match requested configuration")
        cls = {"q_learning": QLearningAgent,"sarsa": SarsaAgent}[meta["algorithm"]]
        agent = cls(TrainingConfig(**meta["training"]),env_config)
        if data["q"].shape != (N_STATES,4) or data["visits"].shape != (N_STATES,4):
            raise ValueError("Incompatible table dimensions")
        if not np.isfinite(data["q"]).all():
            raise ValueError("Non-finite Q-values")
        agent.q[:] = data["q"]; agent.visits[:] = data["visits"]
        agent.ensemble_members = meta.get("ensemble_members",[])
        return agent

def mean_q_ensemble(agents):
    """Average independent learned tables; retain explicit ensemble provenance.

    This is a variance-reduction policy variant, not an additional TD update
    equation and not an individual standard Q-learning or SARSA run.
    """
    if len(agents)<2:
        raise ValueError("At least two independently trained policies required")
    first=agents[0]
    if len({a.config.seed for a in agents})!=len(agents):
        raise ValueError("Ensemble training seeds must be distinct")
    for agent in agents:
        if type(agent) is not type(first) or agent.env_config!=first.env_config:
            raise ValueError("Ensemble members must share algorithm and environment")
        left=asdict(agent.config); right=asdict(first.config)
        left.pop("seed"); right.pop("seed")
        if left!=right: raise ValueError("Ensemble members must share training settings")
    result=type(first)(first.config,first.env_config)
    result.q[:]=np.mean([a.q for a in agents],axis=0)
    result.visits[:]=np.sum([a.visits for a in agents],axis=0)
    result.ensemble_members=[{"algorithm":a.algorithm,"seed":a.config.seed,"episodes":a.config.episodes} for a in agents]
    return result
