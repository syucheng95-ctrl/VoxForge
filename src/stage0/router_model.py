import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class RouterHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        num_categories: int = 14,
        num_tightness: int = 3,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.category_head = nn.Linear(hidden_dim, num_categories)
        self.tightness_head = nn.Linear(hidden_dim, num_tightness)

    def forward(self, embeddings: torch.Tensor) -> dict:
        x = self.input_norm(embeddings)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return {
            "category_logits": self.category_head(x),
            "tightness_logits": self.tightness_head(x),
        }
