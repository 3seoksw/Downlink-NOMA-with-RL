import torch
import torch.nn as nn
import numpy as np

from model.base_model import BaseModel


class ANN(BaseModel):
    """Attention-based Neural Network"""

    def __init__(
        self,
        input_dim=2,
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

        self.prev_state = torch.zeros(batch_size, self.N * self.K, input_dim)
        self.mask = torch.zeros(batch_size, self.N * self.K, dtype=torch.bool)
        self.visited_states = torch.zeros(batch_size, self.N * self.K, dtype=torch.bool)

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

    # NOTE: The shape of the `input` is (batch_size, sequence_length, input_dim).
    def forward(self, input):
        # Find the most recent state
        comp_input = input[:, :, -1]
        diff_mask = torch.ne(self.visited_states, comp_input)
        prev_state_idx = torch.nonzero(diff_mask)
        batch_indices = prev_state_idx[:, 0]
        state_indices = prev_state_idx[:, 1]
        self.prev_state.copy_(input)

        # Update masking
        # self.visited_states[batch_indices, state_indices] = True
        # self.update_mask(batch_indices, state_indices)
        visited_states_clone = self.visited_states.clone()
        visited_states_clone[batch_indices, state_indices] = True
        self.visited_states = visited_states_clone  # Avoid in-place operation
        self.update_mask(batch_indices, state_indices)

        # Pre-encdoer
        pre_encoder = self.pre_encoder_linear(input)

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
            output = masked.masked_fill(self.mask, float("-inf"))
        elif self.method == "Policy Gradient":
            masked = decoder_mul.masked_fill(self.mask, float("-inf"))
            output = nn.functional.softmax(masked, dim=1)
        else:
            raise KeyError("Method is either `Q-Learning` or `Policy Gradient`.")

        return output

    def update_mask(self, batch_indices, state_indices):
        channel_indices = state_indices // self.N
        user_indices = state_indices % self.N

        # User masking
        mask_clone = self.mask.clone()
        for batch_idx, user_idx, channel_idx in zip(
            batch_indices, user_indices, channel_indices
        ):
            channel_indices = torch.arange(self.K, device=self.mask.device)
            state_indices = user_idx + channel_indices * self.N
            mask_clone[batch_idx, state_indices] = True

        # Channel masking
        state_matrix = self.prev_state.view(self.batch_size, self.K, self.N, -1)
        assigned_counts = state_matrix.sum(dim=-1).bool().sum(dim=-1)
        full_channels = (assigned_counts >= 2).nonzero(as_tuple=True)
        for batch_idx, channel_idx in zip(*full_channels):
            state_indices = (
                torch.arange(self.N, device=self.mask.device) + channel_idx * self.N
            )
            mask_clone[batch_idx, state_indices] = True

        mask_clone[self.visited_states] = True
        self.mask = mask_clone  # Avoid in-place operation

        channel_indices = state_indices // self.N
        user_indices = state_indices % self.N

        # User masking
        # for batch_idx, user_idx, channel_idx in zip(
        #     batch_indices, user_indices, channel_indices
        # ):
        #     channel_indices = torch.arange(self.K, device=self.mask.device)
        #     state_indices = user_idx + channel_indices * self.N
        #     self.mask[batch_idx, state_indices] = True

        # # Channel masking
        # state_matrix = self.prev_state.view(self.batch_size, self.K, self.N, -1)
        # assigned_counts = state_matrix.sum(dim=-1).bool().sum(dim=-1)
        # full_channels = (assigned_counts >= 2).nonzero(as_tuple=True)
        # for batch_idx, channel_idx in zip(*full_channels):
        #     state_indices = (
        #         torch.arange(self.N, device=self.mask.device) + channel_idx * self.N
        #     )
        #     self.mask[batch_idx, state_indices] = True

        # self.mask[self.visited_states] = True

    def _reset(self):
        self.prev_state = torch.zeros(self.batch_size, self.N * self.K, self.input_dim)
        self.mask = torch.zeros(self.batch_size, self.N * self.K, dtype=torch.bool)
        self.visited_states = torch.zeros(
            self.batch_size, self.N * self.K, dtype=torch.bool
        )
