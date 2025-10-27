import argparse
from collections import deque

from enviroment import GomokuEnv
from agent import Agent
from replay_buffer import ReplayBuffer
from network import PolicyValueNet
from monte_carlo_tree_search import get_action_from_mcts


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--player_type", type=str, default='local', choices=['local', 'remote'])
    parser.add_argument("--framework", type=str, default='pytorch', choices=['pytorch', 'tensorflow', 'jax'])

    args = parser.parse_args()
    return args

def train(args):
    env = GomokuEnv(board_size=15)
    network = PolicyValueNet(board_size=15, action_size=env.action_size)
    agent = Agent(env, network, args.framework)
    replay_buffer = ReplayBuffer(capacity=10000)

    for episode in range(1000):
        state = env.reset()
        history_frame = deque(maxlen=4)
        history_frame.append(state)
        done = False
        while not done:
            obs
            action, policy = get_action_from_mcts(state, env, agent, n_simulations=100)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, policy, reward)
            
            batch_state, batch_policy, batch_reward = replay_buffer.sample(batch_size=32)
            agent.update_network(batch_state, batch_policy, batch_reward)

            state = next_state
    
    return

def test(args):
    
    return

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    