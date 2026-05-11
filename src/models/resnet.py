"""ResNet-18 adapted for 32×32 CIFAR inputs.

Two modifications from the canonical ImageNet ResNet-18
(He et al., 2016; torchvision reference):

1. The initial 7×7 conv (stride 2) is replaced by a 3×3 conv (stride 1) so
   that spatial resolution is not halved on 32×32 inputs before the first
   residual block.

2. The initial max-pool layer is removed for the same reason.  With both
   changes, the feature map stays at 32×32 through the stem, matching the
   CIFAR convention used in, e.g., He et al.'s own CIFAR experiments.

All other details — BasicBlock with batch-norm, 4 residual stages with
widths [64, 128, 256, 512] and [2, 2, 2, 2] blocks, global average pooling,
and a linear classifier — are unchanged from the standard ResNet-18.

Output: 100 logits (Domain CIFAR-100 benchmark).

Usage:
    from src.models.resnet import ResNet18

    model = ResNet18(cfg.model)
    logits = model(x)   # x: (batch, 3, 32, 32) → logits: (batch, 100)

    # Or build directly (e.g. in tests):
    model = ResNet18.from_dims(input_channels=3, num_classes=100)
"""

from typing import List, Optional, Type

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (two 3×3 convs, batch-norm, skip connection).

    Handles the downsampling shortcut automatically when `stride > 1` or when
    in/out channel counts differ.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        stride:       Stride for the first conv (and the shortcut, if needed).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection when dimensions change
        self.shortcut: Optional[nn.Sequential]
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            identity = self.shortcut(x)

        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

class ResNet18(nn.Module):
    """ResNet-18 with a CIFAR-adapted stem (3×3 conv, no max-pool).

    Args:
        cfg: Model config object (Hydra OmegaConf).  Must expose:
               - input_channels (int): Number of input channels (3 for CIFAR).
               - num_classes     (int): Number of output logits (100 for CIFAR-100).
    """

    # Block counts per stage — identical to canonical ResNet-18
    _STAGE_BLOCKS: List[int] = [2, 2, 2, 2]
    # Channel widths per stage
    _STAGE_WIDTHS: List[int] = [64, 128, 256, 512]

    def __init__(self, cfg) -> None:
        super().__init__()
        self.input_channels: int = cfg.input_channels
        self.num_classes: int = cfg.num_classes

        # ------------------------------------------------------------------
        # Stem: 3×3 conv, stride 1, NO max-pool (CIFAR adaptation)
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels, 64,
                kernel_size=3, stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Residual stages
        # ------------------------------------------------------------------
        self.layer1 = self._make_stage(64,  64,  self._STAGE_BLOCKS[0], stride=1)
        self.layer2 = self._make_stage(64,  128, self._STAGE_BLOCKS[1], stride=2)
        self.layer3 = self._make_stage(128, 256, self._STAGE_BLOCKS[2], stride=2)
        self.layer4 = self._make_stage(256, 512, self._STAGE_BLOCKS[3], stride=2)

        # ------------------------------------------------------------------
        # Head
        # ------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        # Weight initialisation — same as torchvision reference
        self._init_weights()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Stack ``num_blocks`` BasicBlocks into a single stage.

        The first block uses ``stride`` to halve spatial resolution (if > 1)
        and project channels; subsequent blocks use stride 1 throughout.

        Args:
            in_channels:  Input channels for the first block.
            out_channels: Output channels for all blocks in this stage.
            num_blocks:   Number of BasicBlocks.
            stride:       Stride for the first block.

        Returns:
            nn.Sequential containing all blocks.
        """
        blocks: List[nn.Module] = [
            _BasicBlock(in_channels, out_channels, stride=stride)
        ]
        for _ in range(1, num_blocks):
            blocks.append(_BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Kaiming-normal init for conv layers; constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def from_dims(
        cls,
        input_channels: int = 3,
        num_classes: int = 100,
    ) -> "ResNet18":
        """Construct a ResNet18 directly from dimension arguments (e.g. for tests).

        Args:
            input_channels: Number of input channels (default 3).
            num_classes:    Number of output logits (default 100).

        Returns:
            Initialised ResNet18 instance.
        """

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.input_channels = input_channels
        cfg.num_classes = num_classes
        return cls(cfg)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, input_channels, 32, 32)``.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
