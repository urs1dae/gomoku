from click import Tuple
import torch
import numpy as np

from network import PolicyValueNetwork


class Agent:
    def __init__(self, env, args, to_play=0):
        self.env = env
        self.path = args.model_path
        self.to_play = to_play
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network =  PolicyValueNetwork(obs_size=env.state_size, action_size=env.action_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.001, weight_decay=1e-4)

        if args.load_model:
            self.load_model()

    def load_model(self):
        print("Loading model from", self.path)
        self.network.load_state_dict(torch.load(self.path))

    def save_model(self):
        print("Saving model to", self.path)
        torch.save(self.network.state_dict(), self.path)
    
    def get_symmetrized_policy_value(self, raw_obs):
        if raw_obs.dim() == 3:
            obs = raw_obs.unsqueeze(0)
        else:
            obs = raw_obs
            
        B, C, H, W = obs.shape
        
        policy_sum = []
        value_sum = torch.tensor([[0.0]]).to(self.device)

        for k in range(4):
            # Rotation
            rot_obs = torch.rot90(obs, k=k, dims=(2, 3))
        
            policy_logits, value = self.network(rot_obs)
            policy_2d_raw = policy_logits[:, :H * W].reshape(B, H, W)
            policy_original_2d = torch.rot90(policy_2d_raw, k=-k, dims=(1, 2))
            policy_sum.append(policy_original_2d.flatten())
            value_sum += value

            # Flip + Rotation
            flip_rot_obs = torch.flip(rot_obs, dims=[3])
            
            policy_logits_f, value_f = self.network(flip_rot_obs)
            policy_2d_f = policy_logits_f[:, :H * W].reshape(B, H, W)
            policy_unflip = torch.flip(policy_2d_f, dims=[2])
            policy_original_f = torch.rot90(policy_unflip, k=-k, dims=(1, 2))

            policy_sum.append(policy_original_f.flatten())
            value_sum += value_f

        policy_sum_tensor = torch.stack(policy_sum, dim=0).sum(dim=0).expand(B, -1)
        policy_avg_logits = policy_sum_tensor / 8.0
        value_avg = value_sum / 8.0

        return policy_avg_logits, value_avg

    def predict(self, raw_obs, add_noise=False, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        # Use the policy-value network to predict action probabilities and obs value.
        obs, mask = self.prepare_agent_obs(raw_obs)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            # policy, value = self.network(obs)
            policy, value = self.get_symmetrized_policy_value(obs)

            policy.masked_fill_(mask, 1e-9)
            policy = policy / policy.sum(dim=-1, keepdim=True)

        if add_noise:
            valid_mask = ~mask
            num_valid_moves = valid_mask.sum()

            if num_valid_moves > 1:
                concentration = torch.full((num_valid_moves,), dirichlet_alpha).to(self.device)
                
                dirichlet_dist = torch.distributions.Dirichlet(concentration)
                noise = dirichlet_dist.sample()

                noise_vec = torch.zeros_like(policy).to(self.device)
                noise_vec[valid_mask] = noise

                policy = (1 - dirichlet_epsilon) * policy + dirichlet_epsilon * noise_vec
        
        return policy, value
    
    def update_network(self, obs, mask, policy, reward):
        # Update the policy-value network using the provided obs, policy, and reward.
        self.network.train()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        mask = torch.tensor(mask, dtype=torch.bool).to(self.device)
        policy = torch.tensor(policy, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

        policy_pred, value_pred = self.network(obs)
        policy.masked_fill_(mask, 1e-9)
        policy = policy / policy.sum(dim=-1, keepdim=True)

        value_loss = torch.mean((reward - value_pred.squeeze()) ** 2)
        policy_loss = -torch.mean(torch.sum(policy * torch.log(policy_pred + 1e-8), dim=1))

        loss = value_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return value_loss.item(), policy_loss.item()

    def prepare_agent_obs(self, raw_obs, to_play=None):
        # Prepare the obs for the agent (e.g., convert to tensor and add batch dimension).
        if to_play is None:
            to_play = self.to_play

        mask = raw_obs[-1] + raw_obs[-2]
        mask = np.expand_dims(mask.ravel(), axis=0).astype(np.bool_)
        if to_play == 0:
            player_plane = np.zeros(raw_obs.shape[1:])
            even_slices = raw_obs[::2]
            odd_slices = raw_obs[1::2]
            obs = np.concatenate((even_slices, odd_slices, player_plane[None, ...]), axis=0)
            obs = np.expand_dims(obs, axis=0)
        else:
            player_plane = np.ones(raw_obs.shape[1:])
            even_slices = raw_obs[::2]
            odd_slices = raw_obs[1::2]
            obs = np.concatenate((odd_slices, even_slices, player_plane[None, ...]), axis=0)
            obs = np.expand_dims(obs, axis=0)

        return obs, mask
