import torch
import torch.nn as nn

class PolicyValueNetwork(nn.Module):
    def __init__(self, obs_size, action_size, frame_length=8):
        super().__init__()
        self.board_size = obs_size[1]
        self.action_size = action_size
        self.frame_length = frame_length

        self.conv1 = nn.Conv2d(frame_length + 1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.Flatten(),
            nn.Linear(4 * self.board_size * self.board_size, action_size),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value