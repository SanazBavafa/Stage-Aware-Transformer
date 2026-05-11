import math
import torch
from torch import nn

RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class StageAwareTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        conv_size,
        output_dim,
        levels,
        dropconnect=0.0,
        dropout=0.3,
        dropres=0.3,
        num_heads=8,
        num_layers=2,
        ff_dim=None,
        max_len=512,
    ):
        super(StageAwareTransformer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_len = max_len

        if ff_dim is None:
            ff_dim = hidden_dim * 4
        self.input_proj = nn.Linear(input_dim + 1, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        padding = conv_size // 2
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=conv_size,
            padding=padding
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask

    def forward(self, input, time, device):
        """
        input: [B, T, F]
        time : [B, T]
        """
        batch_size, time_step, _ = input.size()

        time_feat = time.unsqueeze(-1)
        x = torch.cat([input, time_feat], dim=-1)

        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        causal_mask = self._causal_mask(time_step, x.device)

        padding_mask = (input.abs().sum(dim=-1) == 0)  # [B, T]

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )  

        residual = x

        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)

        if x_conv.size(1) != residual.size(1):
            x_conv = x_conv[:, :residual.size(1), :]

        x = self.norm(x_conv + residual)
        x = self.dropout(x)

        logits = self.output_layer(x) 
        output = torch.sigmoid(logits)

        distance = torch.zeros(time_step, batch_size, device=device)

        return output, distance