import torch.nn as nn

from utils.mask import get_mask


class FCNN(nn.Module):
    def __init__(
        self,
        num_features=3,
        hidden_dim=128,
        num_users=10,
        num_channels=5,
        device="cpu",
        method="Policy Gradient",
        dropout=0.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.N = num_users
        self.K = num_channels
        input_dim = self.N * self.K * self.num_features
        self.device = device
        self.method = method

        if dropout == 0:
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.N * self.K),
            )
        else:
            print("Dropout Layer Activated")
            self.network = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Flatten(),
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.N * self.K),
            )

    def forward(self, state):
        state = state.to(self.device)
        mask = get_mask(state, self.num_features, self.N, self.K, self.device)

        out = self.network(state)
        output = out.masked_fill(mask, float("-inf"))
        if self.method == "Policy Gradient":
            output = nn.functional.softmax(output, dim=1)

        return output
