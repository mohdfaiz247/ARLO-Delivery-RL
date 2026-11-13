"""Episode training with independently seeded orders, transitions and exploration."""
import numpy as np
import pandas as pd
from .environment import DeliveryMitigationEnv, generate_orders
from .agents import QLearningAgent

def train(agent, progress=None) -> pd.DataFrame:
    config = agent.config
    seeds = np.random.SeedSequence(config.seed).spawn(2)
    order_seed = int(seeds[0].generate_state(1)[0])
    transition_rng = np.random.default_rng(seeds[1])
    orders = generate_orders(config.episodes,order_seed)
    env = DeliveryMitigationEnv(agent.env_config); rows = []
    for episode,order in enumerate(orders):
        epsilon = config.epsilon(episode)
        state,_ = env.reset(int(transition_rng.integers(2**31)),order)
        action = agent.select(state,env.get_valid_actions(),epsilon)
        total = 0.; interventions = 0; changes = 0
        for _ in range(3):
            previous = agent.select(state,env.get_valid_actions())
            next_state,reward,done,_,info = env.step(action)
            next_action = None if done else agent.select(next_state,env.get_valid_actions(),epsilon)
            if isinstance(agent,QLearningAgent):
                agent.update(state,action,reward,next_state,env.get_valid_actions(),done)
            else:
                agent.update(state,action,reward,next_state,next_action,done)
            # Count local greedy changes in the state just updated.
            from .environment import decode, valid_actions
            st,_,_,_,budget,services = decode(state)
            changes += previous != agent.select(state,valid_actions(st,budget,services,env.config.costs))
            total += reward; interventions += action != 0
            state,action = next_state,next_action
            if done:
                break
        rows.append({"episode":episode+1,"return":total,"total_cost":info["total_cost"],
                     "epsilon":epsilon,"interventions":interventions,"policy_changes":changes})
        if progress and (episode % max(1,config.episodes//100) == 0 or episode == config.episodes-1):
            progress((episode+1)/config.episodes)
    frame = pd.DataFrame(rows)
    frame["moving_return"] = frame["return"].rolling(min(250,len(frame)),min_periods=1).mean()
    return frame
