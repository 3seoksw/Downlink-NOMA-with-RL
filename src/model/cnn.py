import torch.nn as nn

from utils.mask import get_mask


class CNN(nn.Module):
    def __init__(
        self,
        num_features=3,
        hidden_dim=128,
        num_users=10,
        num_channels=5,
        device="cpu",
        method="Policy Gradient",
    ):
        super().__init__()
        self.num_features = num_features
        self.N = num_users
        self.K = num_channels
        state_size = self.K * self.N
        self.device = device
        self.method = method

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * state_size, (hidden_dim // 2) * state_size),
            nn.Tanh(),
            nn.Linear((hidden_dim // 2) * state_size, state_size),
        )

    def forward(self, state):
        state = state.to(self.device)

        mask = get_mask(state, self.num_features, self.N, self.K, self.device)
        state = state.transpose(1, 2)
        out = self.net(state)
        output = out.masked_fill(mask, float("-inf"))
        if self.method == "Policy Gradient":
            output = nn.functional.softmax(output, dim=1)

        return output
