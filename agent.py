import torch
import numpy as np

class Agent:
    def __init__(self, env, network, to_play=0):
        self.env = env
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-4)
        self.to_play = to_play  # Current player to play

    def predict(self, raw_obs, no_grad=True, add_noise=False, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        # Use the policy-value network to predict action probabilities and obs value.

        obs, mask = self.prepare_agent_obs(raw_obs)
        obs = torch.tensor(obs, dtype=torch.float32)
        mask = torch.tensor(mask)

        if no_grad:
            with torch.no_grad():
                policy, value = self.network(obs)
        policy, value = self.network(obs)

        policy.masked_fill_(mask, 1e-9)
        policy = policy / policy.sum(dim=-1, keepdim=True)

        if add_noise:
            valid_mask = ~mask
            num_valid_moves = valid_mask.sum()

            if num_valid_moves > 1:
                concentration = torch.full((num_valid_moves,), dirichlet_alpha)
                
                dirichlet_dist = torch.distributions.Dirichlet(concentration)
                noise = dirichlet_dist.sample()

                noise_vec = torch.zeros_like(policy)
                noise_vec[valid_mask] = noise

                policy = (1 - dirichlet_epsilon) * policy + dirichlet_epsilon * noise_vec
        
        return policy, value
    
    def update_network(self, obs, mask, policy, reward):
        # Update the policy-value network using the provided obs, policy, and reward.
        self.network.train()
        obs = torch.tensor(obs, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        policy = torch.tensor(policy, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        policy_pred, value_pred = self.network(obs)

        value_loss = torch.mean((reward - value_pred.squeeze()) ** 2)
        policy_loss = -torch.mean(torch.sum(policy * torch.log(policy_pred + 1e-8), dim=1))
        
        loss = value_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return

    def prepare_agent_obs(self, raw_obs, to_play=None):
        # Prepare the obs for the agent (e.g., convert to tensor and add batch dimension).
        if to_play is None:
            to_play = self.to_play

        mask = raw_obs[-1] + raw_obs[-2]
        mask = np.expand_dims(mask.ravel(), axis=0).astype(np.bool_)
        if to_play == 0:
            obs = np.expand_dims(raw_obs, axis=0)
        else:
            even_slices = raw_obs[::2]
            odd_slices = raw_obs[1::2]

            obs = np.stack((odd_slices, even_slices), axis=1).reshape(1, *raw_obs.shape)

        return obs, mask
