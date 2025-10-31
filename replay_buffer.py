import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, mask, pi, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, mask, pi, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, masks, pis, rewards = zip(*(self.buffer[idx] for idx in indices))
        return np.array(obs), np.array(masks), np.array(pis), np.array(rewards)

    def __len__(self):
        return len(self.buffer)