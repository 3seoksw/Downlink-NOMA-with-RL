import torch.nn as nn

from model.base_model import BaseModel


class DQN(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.online_network = self.build_network(input_dim, hidden_dim, output_dim)
        self.target_network = self.build_network(input_dim, hidden_dim, output_dim)
        self.target_network.load_state_dict(self.online_network.state_dict())

    def forward(self, input, model: str):
        if model == "online":
            return self.online_network(input)
        elif model == "target":
            return self.target_network(input)
        else:
            raise Exception("`online` or `target` available only.")

    def build_network(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.Sigmoid(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(),
        )
