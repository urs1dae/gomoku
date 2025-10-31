import numpy as np
from collections import deque


class GameHistory:
    def __init__(self, maxlen=4):
        self.frames = deque(maxlen=maxlen)

    def append(self, frame):
        self.frames.append(frame)
    
    def get_obs(self):
        return np.concatenate(self.frames, axis=0)

    def get_state(self):
        return self.frames[-1]
    
    def copy(self):
        new_history = GameHistory(self.frames.maxlen)
        for frame in self.frames:
            new_history.append(frame.copy())
        return new_history
