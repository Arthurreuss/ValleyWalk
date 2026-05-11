"""2-hidden-layer MLP for the Rotated MNIST benchmark.

Architecture: input → Linear(784→h1) → ReLU → Linear(h1→h2) → ReLU → Linear(h2→10)

Deliberately has no dropout and no batchnorm so that Hessian-based analysis
(Lanczos, HVP) is as clean as possible — normalisation layers introduce
implicit curvature coupling that complicates Hessian interpretation.

The architecture is configurable from a Hydra config object, defaulting to
the spec's 784 → 400 → 400 → 10 topology.

Usage:
    from src.models.mlp import MLP

    model = MLP(cfg.model)
    logits = model(x)   # x: (batch, 784) → logits: (batch, 10)

    # Or build directly with keyword args (e.g. in tests):
    model = MLP.from_dims(input_size=784, hidden_sizes=[400, 400], num_classes=10)
"""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected MLP with ReLU activations, no regularisation layers.

    Args:
        cfg: Model config object (Hydra OmegaConf).  Must expose:
               - input_size  (int):       Flattened input dimensionality (784 for MNIST).
               - hidden_sizes (list[int]): Width of each hidden layer (e.g. [400, 400]).
               - num_classes  (int):       Number of output logits (10 for MNIST).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.input_size: int = cfg.input_size
        self.hidden_sizes: List[int] = list(cfg.hidden_sizes)
        self.num_classes: int = cfg.num_classes

        self.layers = self._build_layers(
            self.input_size, self.hidden_sizes, self.num_classes
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_layers(
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
    ) -> nn.Sequential:
        """Assemble the layer stack.

        Layout:   Linear → ReLU  (repeated for each hidden layer)
                  Linear          (final classifier, no activation)

        No dropout or batchnorm — keeps the Hessian structure clean.

        Args:
            input_size:   Input feature dimensionality.
            hidden_sizes: List of hidden layer widths.
            num_classes:  Number of output logits.

        Returns:
            nn.Sequential containing all layers.
        """
        layers: List[nn.Module] = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU(inplace=True))
            prev_size = h
        layers.append(nn.Linear(prev_size, num_classes))
        return nn.Sequential(*layers)

    @classmethod
    def from_dims(
        cls,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
    ) -> "MLP":
        """Construct an MLP directly from dimension arguments (e.g. for tests).

        Args:
            input_size:   Input feature dimensionality.
            hidden_sizes: List of hidden layer widths.
            num_classes:  Number of output logits.

        Returns:
            Initialised MLP instance.
        """

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.input_size = input_size
        cfg.hidden_sizes = hidden_sizes
        cfg.num_classes = num_classes
        return cls(cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, input_size)``.
               Must already be flattened (784-dim for MNIST).

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        return self.layers(x)
