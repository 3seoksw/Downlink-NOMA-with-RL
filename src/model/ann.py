import torch
import torch.nn as nn
import numpy as np

from model.base_model import BaseModel


class ANN(nn.Module):
    """Attention-based Neural Network"""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=128,
        batch_size=64,
        num_users=40,
        num_channels=20,
        method="Q-Learning",
        device="cpu",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.N = num_users
        self.K = num_channels
        self.method = method
        self.device = device

        # self.prev_state = torch.zeros(batch_size, self.N * self.K, input_dim, device=self.device)
        self.mask = torch.zeros(batch_size, self.N * self.K, dtype=torch.bool, device=self.device)
        self.visited_states = torch.zeros(batch_size, self.N * self.K, dtype=torch.bool, device=self.device)
        # self.visited_states = torch.zeros(ba)

        # Pre-encoder
        self.pre_encoder_linear = nn.Linear(input_dim, hidden_dim)

        # Encoder (applied the Transformer's structure)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=3)

        # Decoder (Single Head Attention Layer)
        self.decoder_combined = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder_linear = nn.Linear(hidden_dim, hidden_dim)

        self.linear = nn.Linear(self.K * self.N, self.K * self.N)

    # NOTE: The shape of the `state` is (batch_size, sequence_length, input_dim).
    def forward(self, prev_state, state):
        batch_size = state.shape[0]
        prev_state = prev_state.to(self.device)
        state = state.to(self.device)

        # Changed indices
        prev_state_status = prev_state[:, :, -1]
        state_status = state[:, :, -1]
        diff_mask = torch.ne(prev_state_status, state_status)
        prev_state_idx = torch.nonzero(diff_mask)
        batch_indices = prev_state_idx[:, 0]
        state_indices = prev_state_idx[:, 1]

        # Visited indices
        visited_idx = torch.nonzero(state_status)
        visited_batch_indices = visited_idx[:, 0]
        visited_state_indices = visited_idx[:, 1]

        # Update masking
        self.visited_states = torch.zeros(batch_size, self.K * self.N, dtype=torch.bool, device=self.device)
        self.visited_states[visited_batch_indices, visited_state_indices] = True
        mask = self.update_mask(state)

        # Pre-encdoer
        pre_encoder = self.pre_encoder_linear(state)

        # Encoder
        embedding = self.encoder(pre_encoder)  # (batch, length(=NK), hidden)
        last_embedding = embedding[batch_indices, state_indices, :]

        # Pre-decoder
        state_combine = torch.mean(embedding, dim=1)
        concat = torch.cat((state_combine, last_embedding), dim=1)
        concat = concat.unsqueeze(1)  # (batch, 1, hidden * 2)

        # Decoder
        decoder_general_state = self.decoder_combined(concat)  # Q = (1, d^e)
        decoder_states = self.decoder_linear(embedding)  # K = (NK, d^e)

        decoder_mul = torch.matmul(
            decoder_states, decoder_general_state.transpose(1, 2)
        )  # KQ^T = (batch, NK, 1)

        # Masking
        decoder_mul = decoder_mul.squeeze(2)

        if self.method == "Q-Learning":
            masked = self.linear(decoder_mul)
            output = masked.masked_fill(mask, float("-inf"))
        elif self.method == "Policy Gradient":
            masked = decoder_mul.masked_fill(mask, float("-inf"))
            output = nn.functional.softmax(masked, dim=1)
        else:
            raise KeyError("Method is either `Q-Learning` or `Policy Gradient`.")

        return output

    def update_mask(self, state):
        batch_size = state.shape[0]

        # Visited states masking
        mask = torch.zeros(batch_size, self.N * self.K, dtype=torch.bool, device=self.device)
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
            channel_indices = torch.arange(self.K, device=self.mask.device)
            state_indices = user_idx + channel_indices * self.N
            mask[batch_idx, state_indices] = True

        # Channel masking
        state_matrix = state.view(batch_size, self.K, self.N, -1)
        assigned_counts = state_matrix.sum(dim=-1).bool().sum(dim=-1)
        full_channels = (assigned_counts >= 2).nonzero(as_tuple=True)
        for batch_idx, channel_idx in zip(*full_channels):
            state_indices = (
                torch.arange(self.N, device=self.mask.device) + channel_idx * self.N
            )
            mask[batch_idx, state_indices] = True

        return mask

    def _reset(self):
        self.prev_state = torch.zeros(self.batch_size, self.N * self.K, self.input_dim, device=self.device)
        self.mask = torch.zeros(self.batch_size, self.N * self.K, dtype=torch.bool, device=self.device)
        self.visited_states = torch.zeros(
            self.batch_size, self.N * self.K, dtype=torch.bool, device=self.device
        )
