import torch

class Agent:
    def __init__(self, env, network):
        self.env = env
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-4)
    
    def predict(self, state, no_grad=True):
        # Use the policy-value network to predict action probabilities and state value.
        if no_grad:
            with torch.no_grad():
                policy, value = self.network(state)
        policy, value = self.network(state)
        
        mask = (state.sum(dim=1, keepdim=True) == 0).float()
        policy = policy * mask.view(-1)
        policy = policy / policy.sum(dim=-1, keepdim=True)
        
        return policy, value
    
    def update_network(self, state, policy, reward):
        # Update the policy-value network using the provided state, policy, and reward.
        self.network.train()
        
        policy_pred, value_pred = self.network(state)
        
        value_loss = torch.mean((reward - value_pred.squeeze()) ** 2)
        policy_loss = -torch.mean(torch.sum(policy * torch.log(policy_pred + 1e-8), dim=1))
        
        loss = value_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(self, state):
        p, v = self.model(state)
        