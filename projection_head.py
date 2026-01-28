"""
Projection Head for Embedding Learning

Maps backbone features to L2-normalized embedding space for metric learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """
    MLP projection head with L2 normalization.

    Architecture:
        Input (feature_dim) → Hidden (hidden_dim) → Output (embedding_dim)
        Each layer: Linear → BatchNorm → ReLU → Dropout
        Final output: L2-normalized embeddings
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2,
            use_batch_norm: bool = True,
            dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(hidden_dim, output_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))

        self.projection = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"ProjectionHead: {input_dim} → {hidden_dim} → {output_dim} "
            f"({num_layers} layers)"
        )

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Project features to embedding space.

        Args:
            x: Input features [batch_size, input_dim]
            normalize: Whether to L2-normalize output

        Returns:
            embeddings: L2-normalized embeddings [batch_size, output_dim]
        """
        embeddings = self.projection(x)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class SimpleProjection(nn.Module):
    """
    Simpler single-layer projection for very small datasets.
    Use when overfitting is a concern.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        embeddings = self.projection(x)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ResidualProjection(nn.Module):
    """
    Projection head with residual connections for better gradient flow.
    Useful for deeper projection networks.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_blocks: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()

        # Input projection to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        embeddings = self.output_proj(x)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class ResidualBlock(nn.Module):
    """Residual block for projection head."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out, inplace=True)

        return out


def create_projection_head(config) -> nn.Module:
    """
    Factory function to create projection head from config.

    Design principles for few-shot learning:

    1. Moderate capacity (2 layers, 256-512 hidden)
       - Prevents overfitting on 5-20 samples
       - Enough expressiveness for separation

    2. Batch normalization
       - Stabilizes training with small batches
       - Acts as regularization

    3. Dropout (0.1-0.2)
       - Critical regularization for small data
       - Apply before final layer

    4. L2 normalization
       - Ensures all embeddings lie on unit hypersphere
       - Makes cosine similarity equivalent to dot product
       - Prevents magnitude domination
    """
    projection = ProjectionHead(
        input_dim=config.projection.input_dim,
        hidden_dim=config.projection.hidden_dim,
        output_dim=config.projection.output_dim,
        num_layers=config.projection.num_layers,
        use_batch_norm=config.projection.use_batch_norm,
        dropout=config.projection.dropout
    )

    param_count = sum(p.numel() for p in projection.parameters())
    logger.info(f"Projection head parameters: {param_count:,}")

    return projection


class CombinedModel(nn.Module):
    """
    Combined backbone + projection head model.
    Convenient wrapper for end-to-end operations.
    """

    def __init__(self, backbone: nn.Module, projection: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projection = projection

    def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False,
            normalize: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through backbone and projection.

        Args:
            x: Input images [batch_size, 3, H, W]
            return_features: If True, return (embeddings, features)
            normalize: Whether to L2-normalize embeddings

        Returns:
            embeddings: [batch_size, embedding_dim] if not return_features
            (embeddings, features): tuple if return_features=True
        """
        features = self.backbone(x)
        embeddings = self.projection(features, normalize=normalize)

        if return_features:
            return embeddings, features
        return embeddings

    def freeze_backbone(self):
        """Freeze backbone for incremental learning."""
        self.backbone.freeze_backbone()

    def freeze_projection(self):
        """Freeze projection head for incremental learning."""
        for param in self.projection.parameters():
            param.requires_grad = False

    def freeze_all(self):
        """Freeze entire model for inference/incremental."""
        self.freeze_backbone()
        self.freeze_projection()