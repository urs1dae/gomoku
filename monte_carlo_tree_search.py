import torch
import numpy as np


class Node:
    def __init__(self, obs, action, prior_prob, parent=None):
        # Initialize a node in the MCTS tree.
        self.obs = obs
        self.action = action
        self.edges_prob = prior_prob[0].detach().numpy()
        self.edges_visit = np.zeros(prior_prob.shape[1], dtype=int)
        self.edges_value = np.zeros(prior_prob.shape[1], dtype=float)

        self.parent = parent
        self.children = {}
    
    @property
    def value(self):
        # Calculate the average value of the node.
        total_visit = np.sum(self.edges_visit)
        if total_visit == 0:
            return 0
        return self.edges_value.sum() / total_visit

    @property
    def q_value(self):
        # Calculate the Q-value for each action from this node.
        q = np.zeros_like(self.edges_value, dtype=float)
        np.divide(self.edges_value, self.edges_visit, out=q, where=self.edges_visit != 0)
        return q

    def ucb_score(self, c_puct):
        # Calculate the UCB score for action selection.
        total_visit = np.sum(self.edges_visit)
        u = self.q_value + c_puct * self.edges_prob * np.sqrt(total_visit) / (1 + self.edges_visit)
        return u

class MonteCarloTreeSearch:
    def __init__(self, env, agent, game_history, n_simulations, c_puct):
        # Initialize the Monte Carlo Tree Search.
        self.env = env
        self.agent = agent
        self.game_history = game_history
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def search(self):
        # Run MCTS simulations from a given obs to build the search tree.
        obs = self.game_history.get_obs()
        prior_prob, obs_value = self.agent.predict(obs, add_noise=True)
        
        root = Node(obs, action=None, prior_prob=prior_prob)
        to_play = self.env.current_player

        for _ in range(self.n_simulations):
            node = root
            tmp_game_history = self.game_history.copy()
            # print( "Starting new simulation")
            while True:
                state = tmp_game_history.get_state()
                mask = 1 - (state[-1] + state[-2])
                mask = mask.ravel().astype(np.bool_)
                child_action = self.select(node, mask)

                next_state, reward, done = self.env.step(child_action, state, to_play)
                tmp_game_history.append(next_state)
                next_obs = tmp_game_history.get_obs()

                if done or node.children.get(child_action) is None:
                    break
                node = node.children[child_action]

            if done:
                child_value = -reward
            else:
                child_value = self.expand_and_evaluate(node, next_obs, child_action)
            self.backup(node, child_action, child_value)
        return root

    def select(self, node, mask):
        # Select the action with the highest UCB score.
        ucb_scores = node.ucb_score(c_puct=self.c_puct)
        ucb_scores[~mask] = -np.inf
        # print("invalid action:" , np.where(mask==0)[0])
        max_score = np.max(ucb_scores)
        best_actions = np.where(ucb_scores == max_score)[0]
        child_action = np.random.choice(best_actions)
        # print("Selected action:", child_action)
        # if not mask[child_action]:
        #     print("Warning: Selected an invalid action.")
        return child_action

    def expand_and_evaluate(self, node, next_obs, child_action):
        # Expand the tree with a new node and evaluate it.
        child_prior_prob, child_obs_value = self.agent.predict(next_obs)
        node.children[child_action] = Node(next_obs, child_action, 
                                        prior_prob=child_prior_prob, 
                                        parent=node
                                        )
        return child_obs_value

    def backup(self, node, child_action, child_value):
        # Backpropagate the evaluation result up the tree.
        value = - child_value
        while node is not None:
            node.edges_value[child_action] += value
            node.edges_visit[child_action] += 1
            child_action = node.action
            node = node.parent
            value = -value
        return


def get_action_from_mcts(env, agent, game_history, n_simulations=400, c_puct=1.0, temperature=1.0):
    # Get the best action and policy from MCTS search.
    mcts = MonteCarloTreeSearch(env, agent, game_history, n_simulations, c_puct)
    root = mcts.search()
    visit_counts = root.edges_visit
    action_size = len(visit_counts)
    if temperature == 0:
        action_index = np.argmax(visit_counts)
        pi = np.zeros(action_size, dtype=np.float32)
        pi[action_index] = 1.0
    else:
        powered_visits = visit_counts ** (1.0 / temperature)
        pi = powered_visits / np.sum(powered_visits)

    chosen_action = np.random.choice(np.arange(action_size), p=pi)
    
    return chosen_action, pi