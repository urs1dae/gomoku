import torch
import numpy as np


class PureNode:
    def __init__(self, state, action, board_size, parent=None):
        # Initialize a node in the MCTS tree.
        self.state = state
        self.action = action
        self.edges_visit = np.zeros(board_size * board_size, dtype=int)
        self.edges_value = np.zeros(board_size * board_size, dtype=float)

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
        u = self.q_value + c_puct * np.sqrt(total_visit) / (1 + self.edges_visit)
        return u

class PureMonteCarloTreeSearch:
    def __init__(self, env, state, n_simulations, c_puct):
        # Initialize the Monte Carlo Tree Search.
        self.env = env
        self.state = state
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def search(self):
        # Run MCTS simulations from a given state to build the search tree.

        root = PureNode(self.state, action=None, board_size=self.env.board_size)
        to_play = self.env.current_player

        for _ in range(self.n_simulations):
            node = root
            state = self.state.copy()
            while True:
                mask = 1 - (state[-1] + state[-2])
                mask = mask.ravel().astype(np.bool_)
                child_action = self.select(node, mask)

                next_state, reward, done = self.env.step(child_action, state, to_play)
                state = next_state

                if done or node.children.get(child_action) is None:
                    break
                node = node.children[child_action]
                to_play = 1 - to_play

            if done:
                child_value = -reward
            else:
                child_value = self.expand_and_rollout(node, state, child_action, to_play)
            self.backup(node, child_action, child_value)
        return root

    def select(self, node, mask):
        ucb_scores = node.ucb_score(c_puct=self.c_puct)
        ucb_scores[~mask] = -np.inf
        max_score = np.max(ucb_scores)
        best_actions = np.where(ucb_scores == max_score)[0]
        child_action = np.random.choice(best_actions)

        return child_action

    def expand_and_rollout(self, node, state, child_action, child_player):
        # Expand the tree with a new node and evaluate it.
        node.children[child_action] = PureNode(state, child_action, 
                                        board_size=self.env.board_size,
                                        parent=node
                                        )
        
        done = False
        current_state = state
        to_play = child_player
        while not done:
            mask = 1 - (current_state[-1] + current_state[-2])
            mask = mask.ravel().astype(np.bool_)
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)

            next_state, reward, done = self.env.step(action, current_state, to_play)
            if done: break
            
            to_play = 1 - to_play
            current_state = next_state
        reward = reward if to_play == child_player else -reward
        return reward

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


def get_action_from_puremcts(env, state, n_simulations=400, c_puct=1.0, temperature=1.0):
    # Get the best action and policy from MCTS search.
    mcts = PureMonteCarloTreeSearch(env, state, n_simulations, c_puct)
    root = mcts.search()
    visit_counts = root.edges_visit
    action_size = len(visit_counts)
    if temperature == 0.0:
        action_index = np.argmax(visit_counts)
        pi = np.zeros(action_size, dtype=np.float32)
        pi[action_index] = 1.0
    else:
        powered_visits = visit_counts ** (1.0 / temperature)
        pi = powered_visits / np.sum(powered_visits)

    chosen_action = np.random.choice(np.arange(action_size), p=pi)
    
    return chosen_action, pi