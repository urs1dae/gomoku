import swanlab
import argparse

from enviroment import GomokuEnv
from agent import Agent
from replay_buffer import ReplayBuffer
from monte_carlo_tree_search import get_action_from_mcts
from pure_monte_carlo_tree_search import get_action_from_puremcts
from utils import GameHistory


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--player_type", type=str, default='local', choices=['local', 'remote'])
    parser.add_argument("--framework", type=str, default='pytorch', choices=['pytorch', 'tensorflow', 'jax'])
    parser.add_argument("--to_play", type=int, default=0, help="Which player the agent will play as (0 for BLACK, 1 for WHITE)")
    parser.add_argument("--model_path", type=str, default="model/model.pth", help="Path to the model file")
    parser.add_argument("--load_model", default=False, action="store_true", help="Whether to load the model from the specified path")
    parser.add_argument("--opponent", type=str, default="self", choices=["pure_mcts", "human", "self", "remote"], help="Type of opponent to play against during testing")

    args = parser.parse_args()
    return args

def train(args):
    swanlab.init()
    win_counts = [0, 0]

    env = GomokuEnv(board_size=15, is_render=False)
    agent = Agent(env, args, to_play=0)
    replay_buffer = ReplayBuffer(capacity=10_000_000)

    for episode in range(3000):
        print(f"\nEpisode {episode+1} starts.")
        state = env.reset()
        done = False
        episode_steps = []
        turn = 0

        game_history = GameHistory()
        for _ in range(4):
            game_history.append(state)

        while not done:
            turn += 1
            obs = game_history.get_obs()
            action, policy = get_action_from_mcts(env, agent, game_history, n_simulations=400)
            next_state, reward, done = env.step(action)

            obs, mask = agent.prepare_agent_obs(obs)
            game_history.append(next_state)
            episode_steps.append((obs[0], mask[0], policy))
            agent.to_play = 1 - agent.to_play

        reward = 1 if env.winner == 0 else -1
        for step in episode_steps:
            obs, mask, policy = step
            replay_buffer.push(obs, mask, policy, reward)
            reward = -reward

        win_counts[0 if env.winner == 0 else 1] += 1
        print(f"\tTurns: {turn}")
        print(f"\tBlack vs White: {win_counts[0]} vs {win_counts[1]}")
        swanlab.log({"black_vs_white_ratio": win_counts[0] / (win_counts[1] + 1e-5)})

        if len(replay_buffer) > 128:
            for _ in range(100):
                batch_obs, batch_mask, batch_policy, batch_reward = replay_buffer.sample(batch_size=128)
                value_loss, policy_loss = agent.update_network(batch_obs, batch_mask, batch_policy, batch_reward)
                swanlab.log({
                    "loss": value_loss + policy_loss,
                    "value_loss": value_loss,
                    "policy_loss": policy_loss
                })

        if episode % 10 == 0:
            agent.save_model()

    env.close()
    swanlab.finish()

    return

def test(args):

    win_counts = [0, 0]
    env = GomokuEnv(board_size=15, is_render=True)
    agent = Agent(env, args, to_play=0)

    player = "pure_mcts"

    print(f"Black is alphazero, White is {player}")

    for episode in range(10):
        print(f"\nEpisode {episode+1} starts.")

        turn = 1
        state = env.reset()
        done = False

        game_history = GameHistory()
        for _ in range(4):
            game_history.append(state)

        while not done:
            print(f"\tTurn {turn}: Player {'BLACK' if env.current_player == 0 else 'WHITE'}'s move")
            if env.current_player == agent.to_play:
                action, _ = get_action_from_mcts(env, agent, game_history, n_simulations=400, temperature=0.0)
            else:
                action, _ = get_action_from_puremcts(env, state, n_simulations=400, temperature=0.0)
    
            state, _, done = env.step(action)

            if done:
                win_counts[0 if env.winner == agent.to_play else 1] += 1
                print(f"\tTurns: {turn}")
                print(f"\tWinner: {'alphazero' if env.winner == agent.to_play else player }")
                print(f"\tBlack vs White: {win_counts[0]} vs {win_counts[1]}")
                break

            turn += 1
            game_history.append(state)

    env.close()

    return

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    