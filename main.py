import argparse
import numpy as np
from collections import deque

from enviroment import GomokuEnv
from agent import Agent
from replay_buffer import ReplayBuffer
from network import PolicyValueNetwork
from monte_carlo_tree_search import get_action_from_mcts
from utils import GameHistory


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--player_type", type=str, default='local', choices=['local', 'remote'])
    parser.add_argument("--framework", type=str, default='pytorch', choices=['pytorch', 'tensorflow', 'jax'])
    parser.add_argument("--to_play", type=int, default=0, help="Which player the agent will play as (0 for BLACK, 1 for WHITE)")

    args = parser.parse_args()
    return args

def train(args):
    env = GomokuEnv(board_size=15)
    network = PolicyValueNetwork(obs_size=env.state_size, action_size=env.action_size)
    agent = Agent(env, network, to_play=0)
    replay_buffer = ReplayBuffer(capacity=10000)

    for episode in range(1000):
        state = env.reset()

        game_history = GameHistory()
        for _ in range(4):
            game_history.append(state)

        done = False
        while not done:
            obs = game_history.get_obs()
            action, policy = get_action_from_mcts(env, agent, game_history, n_simulations=100)
            next_state, reward, done = env.step(action)
            game_history.append(next_state)

            obs, mask = agent.prepare_agent_obs(obs)
            replay_buffer.push(obs[0], mask[0], policy, reward)

            if len(replay_buffer) >= 32:
                batch_obs, batch_mask, batch_policy, batch_reward = replay_buffer.sample(batch_size=32)
                agent.update_network(batch_obs, batch_mask, batch_policy, batch_reward)

            agent.to_play = 1 - agent.to_play

    return

def test(args):
    
    return

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    