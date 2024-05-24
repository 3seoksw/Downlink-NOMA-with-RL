import torch
import torch.nn as nn
import numpy as np

from model.base_model import BaseModel


class ANN(BaseModel):
    """Attention-based Neural Network"""

    def __init__(
        self, input_dim=2, hidden_dim=128, batch_size=64, num_users=40, num_channels=20
    ):
        super().__init__()
        self.N = num_users
        self.K = num_channels
        self.prev_state = torch.zeros(batch_size, self.N * self.K, 2)
        self.prev_state_idx = 0
        self.mask = torch.zeros(batch_size, self.N * self.K, dtype=torch.bool)

        # Pre-encoder
        self.pre_encoder_linear = nn.Linear(input_dim, hidden_dim)

        # Encoder (applied the Transformer's structure)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)

        # Decoder (Single Head Attention Layer)
        self.decoder_combined = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder_linear = nn.Linear(hidden_dim, hidden_dim)

    # NOTE: The shape of the `input` is (batch_size, sequence_length, input_dim).
    def forward(self, input):
        # Find the most recent state
        diff_mask = torch.ne(self.prev_state, input)
        diff_mask = diff_mask.any(dim=-1)
        prev_state_idx = torch.nonzero(diff_mask)
        batch_indices = prev_state_idx[:, 0]
        state_indices = prev_state_idx[:, 1]
        self.prev_state = input

        # Update masking
        self.mask[batch_indices, state_indices] = True

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
        masked = decoder_mul.masked_fill(self.mask, float("-inf"))

        compatibility = nn.functional.softmax(masked, dim=1)

        return compatibility

    # WARN: Deprecated
    def get_mask(self, input):
        user_idx = input[:, :, 0]
        channel_idx = input[:, :, 1]
        values = channel_idx * self.N + user_idx
        values = values.to(torch.int64)

        visited_states = torch.nonzero(input)
        # visited_states = visited_states.any(dim=-1)

        # Mask
        mask = torch.ones(input.shape[0], self.N * self.K)
        for i, batch in enumerate(values):
            channel_counts = torch.zeros(self.K)
            for n in batch:  # Traverse through history states and filter out
                if n < 0:
                    break
                # History filter
                mask[i][n] = 0

                # User filter
                channel_idx = n // self.N
                user_idx = n - channel_idx * self.N
                for j in range(self.K):
                    mask[i][self.N * j + user_idx] = 0

                # Channel filter
                channel_counts[channel_idx] += 1
                if channel_counts[channel_idx] >= 2:
                    for j in range(self.N):
                        mask[i][self.N * channel_idx + j] = 0

        mask = mask.unsqueeze(-1)
        return mask.bool()
