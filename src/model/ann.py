import torch
import torch.nn as nn

from model.base_model import BaseModel


class ANN(BaseModel):
    """Attention-based Neural Network"""

    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1):
        super().__init__()

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
        embedding = self.encoder(pre_encoder)
        last_embedding = embedding[:, -1, :]

        # Pre-decoder
        state_combine = torch.mean(embedding, dim=1)
        concat = torch.cat((state_combine, last_embedding), dim=1)
        concat = concat.unsqueeze(1)

        # Decoder
        decoder_combined = self.decoder_combined(concat)
        decoder_linear = self.decoder_linear(embedding)
        decoder_mul = torch.matmul(decoder_combined, decoder_linear.transpose(1, 2))
        decoder_mul = decoder_mul.squeeze(1)

        return decoder_mul
