import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        num_features=3,
        hidden_dim=128,
        num_users=40,
        num_channels=20,
        device="cpu",
    ):
        super().__init__()
        self.N = num_users
        self.K = num_channels
        state_size = self.K * self.N
        self.device = device

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=state_size,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * num_features, 64 * num_features),
            nn.ReLU(),
            nn.Linear(64 * num_features, state_size),
        )

    def forward(self, state):
        state = state.to(self.device)

        mask = self.get_mask(state)
        out = self.net(state)
        output = out.masked_fill(mask, float("-inf"))

        return output

    def get_mask(self, state):
        batch_size = state.shape[0]

        # Visited indices
        state_status = state[:, :, -1]
        visited_idx = torch.nonzero(state_status)
        visited_batch_indices = visited_idx[:, 0]
        visited_state_indices = visited_idx[:, 1]

        # Update masking
        self.visited_states = torch.zeros(
            batch_size, self.K * self.N, dtype=torch.bool, device=self.device
        )
        self.visited_states[visited_batch_indices, visited_state_indices] = True

        mask = torch.zeros(
            batch_size, self.N * self.K, dtype=torch.bool, device=self.device
        )
        mask[self.visited_states] = True

        indices = torch.nonzero(self.visited_states)
        batch_indices = indices[:, 0]
        state_indices = indices[:, 1]
        channel_indices = state_indices // self.N
        user_indices = state_indices % self.N

        # User masking
        for batch_idx, user_idx, channel_idx in zip(
            batch_indices, user_indices, channel_indices
        ):
            channel_indices = torch.arange(self.K, device=self.device)
            state_indices = user_idx + channel_indices * self.N
            mask[batch_idx, state_indices] = True

        # Channel masking
        state_status = state[:, :, 2]
        state_matrix = state_status.view(batch_size, self.K, self.N, -1)
        assigned_counts = state_matrix.sum(dim=-1).bool().sum(dim=-1)
        full_channels = (assigned_counts >= 2).nonzero(as_tuple=True)
        for batch_idx, channel_idx in zip(*full_channels):
            state_indices = (
                torch.arange(self.N, device=self.device) + channel_idx * self.N
            )
            mask[batch_idx, state_indices] = True

        return mask
