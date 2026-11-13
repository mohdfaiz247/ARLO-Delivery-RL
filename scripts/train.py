"""Run from the project root: python scripts/train.py --algorithm q_learning."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from delivery_rl.agents import QLearningAgent,SarsaAgent,TrainingConfig
from delivery_rl.environment import EnvConfig
from delivery_rl.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm",choices=["q_learning","sarsa"],default="q_learning")
    parser.add_argument("--episodes",type=int,default=15000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--scenario",choices=["moderate","pessimistic","optimistic"],default="moderate")
    args = parser.parse_args()
    cls = QLearningAgent if args.algorithm == "q_learning" else SarsaAgent
    agent = cls(TrainingConfig(episodes=args.episodes,seed=args.seed),EnvConfig(args.scenario))
    curve = train(agent)
    agent.save(f"models/{args.algorithm}.npz")
    Path("results").mkdir(exist_ok=True)
    curve.to_csv(f"results/{args.algorithm}_training.csv",index=False)
    print(f"Saved {args.algorithm}: {args.episodes:,} episodes; final moving return {curve.moving_return.iloc[-1]:.2f}")
