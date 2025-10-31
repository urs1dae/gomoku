import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0        

    def push(self, raw_obs, raw_mask, pi, raw_reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (raw_obs, raw_mask, pi, raw_reward)
        self.position = (self.position + 1) % self.capacity

        C, H, W = raw_obs.shape
        A = raw_mask.shape[0]
        pi_2d = pi[:A].reshape(H, W)
        mask_2d = raw_mask[:A].reshape(H, W)

        for k in range(4):            
            # Rotation
            rot_obs = np.rot90(raw_obs, k=k, axes=(1, 2))
            rot_pi_2d = np.rot90(pi_2d, k=k)
            rot_mask_2d = np.rot90(mask_2d, k=k)
            self.buffer.append((rot_obs, rot_mask_2d.reshape(-1), rot_pi_2d.reshape(-1), raw_reward))
            self.position = (self.position + 1) % self.capacity

            # Flip + Rotation
            flip_rot_obs = np.flip(rot_obs, axis=2)
            flip_rot_pi_2d = np.flip(rot_pi_2d, axis=1)
            flip_rot_mask_2d = np.flip(rot_mask_2d, axis=1)

            self.buffer.append((flip_rot_obs, flip_rot_mask_2d.reshape(-1), flip_rot_pi_2d.reshape(-1), raw_reward))
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, masks, pis, rewards = zip(*(self.buffer[idx] for idx in indices))
        return np.array(obs), np.array(masks), np.array(pis), np.array(rewards)

    def __len__(self):
        return len(self.buffer)