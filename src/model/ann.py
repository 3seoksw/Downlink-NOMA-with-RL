import torch
import torch.nn as nn
import numpy as np

from model.base_model import BaseModel


class ANN(BaseModel):
    """Attention-based Neural Network"""

    def __init__(self, input_dim=2, hidden_dim=128, num_users=10, num_channels=15):
        super().__init__()
        self.N = num_users
        self.K = num_channels

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
        # Pre-encdoer
        pre_encoder = self.pre_encoder_linear(input)

        # Encoder
        embedding = self.encoder(pre_encoder)  # (batch, length(=NK), hidden)
        last_embedding = embedding[:, -1, :]
        print(f"EMBED: {embedding.shape}")

        # Pre-decoder
        state_combine = torch.mean(embedding, dim=1)
        concat = torch.cat((state_combine, last_embedding), dim=1)
        concat = concat.unsqueeze(1)  # (batch, 1, hidden * 2)

        # Decoder
        decoder_general_state = self.decoder_combined(concat)  # Q = e^d \times W^Q
        decoder_states = self.decoder_linear(embedding)  # K = E^s \times W^K

        decoder_mul = torch.matmul(
            decoder_states, decoder_general_state.transpose(1, 2)
        )  # KQ^T =(batch, NK, 1)

        # Masking
        mask = self.get_mask(input)
        indices = np.where(mask == 1)
        batch_indices = indices[0]
        available_states_indices = indices[1]

        masked = decoder_mul[mask]

        compatibility = nn.functional.softmax(masked, dim=0)

        output = torch.zeros(input.shape[0], self.K * self.N, dtype=torch.float32)
        output[batch_indices, available_states_indices] = compatibility
        # output = output.sigmoid()

        return output

    def get_mask(self, input):
        user_idx = input[:, :, 0]
        channel_idx = input[:, :, 1]
        values = channel_idx * self.N + user_idx
        values = values.to(torch.int64)

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
