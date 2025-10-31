import swanlab
import argparse

from enviroment import GomokuEnv
from agent import Agent
from replay_buffer import ReplayBuffer
from monte_carlo_tree_search import get_action_from_mcts
from utils import GameHistory


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--player_type", type=str, default='local', choices=['local', 'remote'])
    parser.add_argument("--framework", type=str, default='pytorch', choices=['pytorch', 'tensorflow', 'jax'])
    parser.add_argument("--to_play", type=int, default=0, help="Which player the agent will play as (0 for BLACK, 1 for WHITE)")
    parser.add_argument("--model_path", type=str, default="model/model.pth", help="Path to the model file")
    parser.add_argument("--load_model", default=False, action="store_true", help="Whether to load the model from the specified path")

    args = parser.parse_args()
    return args

def train(args):
    swanlab.init()
    win_counts = [1, 1]

    env = GomokuEnv(board_size=15)
    agent = Agent(env, args, to_play=0)
    replay_buffer = ReplayBuffer(capacity=10_000_000)

    for episode in range(100_000):
        state = env.reset()
        episode_steps = []

        game_history = GameHistory()
        for _ in range(4):
            game_history.append(state)

        done = False
        while not done:
            obs = game_history.get_obs()
            action, policy = get_action_from_mcts(env, agent, game_history, n_simulations=200)
            next_state, reward, done = env.step(action)

            obs, mask = agent.prepare_agent_obs(obs)
            game_history.append(next_state)
            episode_steps.append((obs[0], mask[0], policy))
            agent.to_play = 1 - agent.to_play

            if len(replay_buffer) > 32:
                batch_obs, batch_mask, batch_policy, batch_reward = replay_buffer.sample(batch_size=32)
                value_loss, policy_loss = agent.update_network(batch_obs, batch_mask, batch_policy, batch_reward)
                swanlab.log({
                    "loss": value_loss + policy_loss,
                    "value_loss": value_loss,
                    "policy_loss": policy_loss
                })

            if done:
                reward = 1 if env.current_player == 0 else -1
                win_counts[0 if reward == 1 else 1] += 1
                for step in episode_steps:
                    obs, mask, policy = step
                    replay_buffer.push(obs, mask, policy, reward)
                    reward = -reward
                swanlab.log({"black_vs_white_ratio": win_counts[0] / (win_counts[1])})

        if episode % 100 == 0:
            agent.save_model()

    env.close()
    swanlab.finish()

    return

def test(args):
    
    return

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    