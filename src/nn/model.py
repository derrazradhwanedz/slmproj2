"""Dense neural network mapping a question's quality profile to a target answer quality profile."""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from metrics import profile as profile_metrics
from utils import load_yaml

NUM_METRICS: int = len(profile_metrics.__all__)
_MODEL_DEFAULTS = load_yaml("nn.yaml")["model"]


@dataclass(frozen=True)
class QAPredictorConfig:
    """Architecture hyperparameters for QAPredictorNN.

    Attributes:
        num_metrics: Input/output dimensionality. Defaults to the number of
            metrics currently registered in app.src.metrics.profile, so the
            model stays in sync if metrics are added or removed.
        hidden_dims: Size of each hidden layer, in order.
        dropout_rates: Dropout probability applied after each hidden layer.
            Must be the same length as hidden_dims.
        use_batchnorm: Whether to apply BatchNorm1d after each hidden layer.
            Must be the same length as hidden_dims.
    """

    num_metrics: int = NUM_METRICS
    hidden_dims: List[int] = field(default_factory=lambda: list(_MODEL_DEFAULTS["hidden_dims"]))
    dropout_rates: List[float] = field(default_factory=lambda: list(_MODEL_DEFAULTS["dropout_rates"]))
    use_batchnorm: List[bool] = field(default_factory=lambda: list(_MODEL_DEFAULTS["use_batchnorm"]))

    def __post_init__(self) -> None:
        """Validate that per-layer config lists agree in length.

        Raises:
            ValueError: If hidden_dims, dropout_rates, and use_batchnorm
                don't all have the same length.
        """
        lengths = {len(self.hidden_dims), len(self.dropout_rates), len(self.use_batchnorm)}
        if len(lengths) != 1:
            raise ValueError(
                "hidden_dims, dropout_rates, and use_batchnorm must all have "
                f"the same length, got {lengths}"
            )


class QAPredictorNN(nn.Module):
    """Feedforward network: question quality profile -> target answer quality profile."""

    def __init__(self, config: Optional[QAPredictorConfig] = None) -> None:
        """Build the network from a config.

        Args:
            config: Architecture hyperparameters. Defaults to QAPredictorConfig().
        """
        super().__init__()
        self.config = config or QAPredictorConfig()
        self.num_metrics = self.config.num_metrics

        layers: List[nn.Module] = []
        in_dim = self.config.num_metrics
        for hidden_dim, dropout, batchnorm in zip(
            self.config.hidden_dims, self.config.dropout_rates, self.config.use_batchnorm
        ):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.config.num_metrics))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the target answer quality profile.

        Args:
            x: Tensor of shape (batch_size, num_metrics) with normalized
                question quality metrics.

        Returns:
            Tensor of shape (batch_size, num_metrics) with predicted
            (normalized) answer quality metrics.

        Raises:
            ValueError: If the last dimension of x does not match num_metrics.
        """
        if x.shape[-1] != self.num_metrics:
            raise ValueError(
                f"Expected input of size {self.num_metrics}, got {x.shape[-1]}"
            )
        return self.net(x)
