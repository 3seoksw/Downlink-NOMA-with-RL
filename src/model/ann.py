import torch
import torch.nn as nn

from utils.mask import get_mask


class ANN(nn.Module):
    """Attention-based Neural Network"""

    def __init__(
        self,
        num_features=3,
        hidden_dim=128,
        num_users=10,
        num_channels=5,
        method="Policy Gradient",
        device="cpu",
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.N = num_users
        self.K = num_channels
        self.method = method
        self.device = device

        # self.prev_state = torch.zeros(batch_size, self.N * self.K, input_dim, device=self.device)
        # Pre-encoder
        self.pre_encoder_linear = nn.Linear(num_features, hidden_dim)

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
    def forward(self, state):
        state = state.to(self.device)

        # Update masking
        # mask = self.get_mask(state)
        mask = get_mask(state, self.num_features, self.N, self.K, self.device)

        # Pre-encdoer
        pre_encoder = self.pre_encoder_linear(state)

        # Encoder
        embedding = self.encoder(pre_encoder)  # (batch, length(=NK), hidden)
        last_embedding = embedding[:, -1, :]

        # Pre-decoder
        state_combine = torch.sum(embedding, dim=1)
        concat = torch.cat((state_combine, last_embedding), dim=1)
        concat = concat.unsqueeze(1)  # (batch, 1, hidden * 2)

        # Decoder
        decoder_general_state = self.decoder_combined(concat)  # Q = (1, d^e)
        # decoder_general_state = decoder_general_state.unsqueeze(0)
        decoder_states = self.decoder_linear(embedding)  # K = (NK, d^e)

        decoder_mul = torch.matmul(
            decoder_states, decoder_general_state.transpose(1, 2)
        )  # KQ^T = (batch, NK, 1)

        # Masking
        decoder_mul = decoder_mul.squeeze(2)

        if self.method == "Q-Learning":
            masked = self.linear(decoder_mul)
            output = masked.masked_fill(mask, -100)
        elif self.method == "Policy Gradient":
            masked = decoder_mul.masked_fill(mask, float("-inf"))
            output = nn.functional.softmax(masked, dim=1)
        else:
            raise KeyError("Method is either `Q-Learning` or `Policy Gradient`.")

        return output
