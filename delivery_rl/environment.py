"""A finite, fully observable MDP with stage-indexed common random numbers."""
from dataclasses import dataclass, asdict
from math import exp
import numpy as np

ACTION_NAMES = ("No intervention", "Priority processing", "Expedited transport", "Combined intervention")
STAGES = ("Approved", "Preparing", "Dispatched", "Delivered")
DISTANCES = ("short", "medium", "long")
STATE_DIMS = (3, 12, 2, 3, 51, 4)
N_STATES = int(np.prod(STATE_DIMS))
ENCODING = "v1:stage,outlook_plus3,load,distance,exact_budget,service_bits"

@dataclass(frozen=True)
class Scenario:
    name: str
    priority_success: float
    transport_success: float
    combined_success: float
    improvement_scale: float
    disruption: float

SCENARIOS = {
    "pessimistic": Scenario("pessimistic", .45, .60, .70, .75, .32),
    "moderate": Scenario("moderate", .65, .80, .90, 1., .18),
    "optimistic": Scenario("optimistic", .80, .92, .97, 1.25, .08),
}

@dataclass(frozen=True)
class EnvConfig:
    scenario: str = "moderate"
    costs: tuple = (0, 8, 25, 32)
    cost_per_late_day: float = 15.
    fixed_late_penalty: float = 20.

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError("Unknown scenario")
        if len(self.costs) != 4 or self.costs[0] != 0 or any(int(c) != c or c < 0 for c in self.costs):
            raise ValueError("Four nonnegative integer costs required; no action must cost zero")
        if self.cost_per_late_day < 0 or self.fixed_late_penalty < 0:
            raise ValueError("Penalties must be nonnegative")

def risk(outlook: int, distance: int, load: int) -> float:
    """Observable disruption pressure, not a calibrated real-world probability."""
    return 1 / (1 + exp(-(outlook + .25 * distance + .35 * load - .5) / 1.7))

@dataclass(frozen=True)
class Order:
    order_id: str
    outlook: int
    distance: int
    load: int
    initial_budget: int

    def __post_init__(self):
        for value, low, high in ((self.outlook,-3,8),(self.distance,0,2),(self.load,0,1),(self.initial_budget,0,50)):
            if not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError("Order fields must be bounded integers")

    def record(self):
        return {**asdict(self), "initial_delay_risk": risk(self.outlook,self.distance,self.load),
                "expected_late_days": max(0,self.outlook), "distance_label": DISTANCES[self.distance]}

def generate_orders(n: int, seed: int, balanced: bool = False) -> list[Order]:
    rng = np.random.default_rng(seed)
    orders = []
    for i in range(n):
        distance = int(rng.integers(3)); load = int(rng.integers(2))
        outlook = int(rng.integers(-3,9)) if balanced else int(np.clip(np.rint(rng.normal(1+.65*distance+.7*load,2.2)),-3,7))
        budget = int(rng.choice([0,8,25,32,40,50]))
        orders.append(Order(f"S{seed}-{i:05d}",outlook,distance,load,budget))
    return orders

def valid_actions(stage: int, budget: int, services: int, costs=(0,8,25,32)) -> list[int]:
    if stage >= 3:
        return []
    candidates = ((0,1,3),(0,1,2),(0,2))[stage]
    return [a for a in candidates if costs[a] <= budget and
            not (a == 1 and services & 1) and not (a == 2 and services & 2) and
            not (a == 3 and services)]

def decode(state: int) -> tuple:
    return tuple(int(v) for v in np.unravel_index(state,STATE_DIMS))

class DeliveryMitigationEnv:
    """Three decisions. Rewards telescope exactly to negative economic cost.

    Outlook is a signed integer-day trajectory, including early-delivery slack.
    Each stage adds a bounded disruption or recovery. Successful priority acts
    immediately; successful transport acts once at dispatch and persists as a
    purchased service. Future randomness is never part of the observation.
    """
    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.scenario = SCENARIOS[self.config.scenario]
        self.stage = 3

    def reset(self, seed: int | None = None, order: Order | None = None):
        rng = np.random.default_rng(seed)
        self.order = order or generate_orders(1,int(rng.integers(2**31)))[0]
        self.stage = 0; self.outlook = self.order.outlook
        self.budget = self.order.initial_budget; self.services = 0
        self.spent = 0.; self.delay_cost = 0.; self.history = []
        # Fixed stage/channel draws: policy action choices never shift disruptions.
        self.draws = rng.random((3,5))
        return self.get_state(), self.info()

    def get_state(self) -> int | None:
        if self.stage == 3:
            return None
        return int(np.ravel_multi_index((self.stage,self.outlook+3,self.order.load,
                   self.order.distance,self.budget,self.services),STATE_DIMS))

    def get_valid_actions(self) -> list[int]:
        return valid_actions(self.stage,self.budget,self.services,self.config.costs)

    def info(self) -> dict:
        return {"stage": self.stage, "outlook": self.outlook,
                "risk": risk(self.outlook,self.order.distance,self.order.load),
                "expected_late_days": max(0,self.outlook), "remaining_budget": self.budget,
                "services": self.services, "intervention_cost": self.spent,
                "delay_cost": self.delay_cost, "total_cost": self.spent+self.delay_cost,
                "late_days": max(0,self.outlook), "valid_actions": self.get_valid_actions()}

    def step(self, action: int):
        if action not in self.get_valid_actions():
            raise ValueError(f"Invalid action {action} at stage {self.stage}")
        before = self.info(); stage = self.stage; d = self.draws[stage]; s = self.scenario
        cost = self.config.costs[action]; self.budget -= cost; self.spent += cost
        saved = 0; success = None
        if action in (1,3):
            self.services |= 1
            p = s.combined_success if action == 3 else s.priority_success
            success = bool(d[1] < p)
            whole = int(s.improvement_scale)
            saved += whole + int(d[4] < s.improvement_scale-whole) if success else 0
        if action in (2,3):
            self.services |= 2
        if stage == 2 and self.services & 2:
            # Combined service uses its advertised reliability at dispatch.
            # Bit 1 identifies priority as well; both purchased separately also qualify.
            p = s.combined_success if self.services & 1 else s.transport_success
            transport_success = bool(d[2] < p)
            saved += int(np.clip(round((1+int(d[3]*3))*s.improvement_scale),1,4)) if transport_success else 0
            success = transport_success
        disturbance_probability = min(.85,s.disruption + .10*self.order.load + (.07*self.order.distance if stage == 2 else 0))
        disturbance = (1 + int(d[3] > .8)) if d[0] < disturbance_probability else (-1 if d[0] > .92 else 0)
        self.outlook = int(np.clip(self.outlook + disturbance - saved,-3,8))
        self.stage += 1
        terminated = self.stage == 3
        if terminated:
            days = max(0,self.outlook)
            self.delay_cost = days*self.config.cost_per_late_day + (self.config.fixed_late_penalty if days else 0)
        reward = -float(cost) - (self.delay_cost if terminated else 0)
        info = self.info()
        self.history.append({"decision_stage": STAGES[stage], "action": ACTION_NAMES[action],
            "action_id": action, "action_cost": cost, "success": success, "saved_days": saved,
            "disturbance_days": disturbance, "before_risk": before["risk"],
            "before_outlook": before["outlook"], "reward": reward, **info})
        return self.get_state(), reward, terminated, False, info
