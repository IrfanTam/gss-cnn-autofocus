"""
Custom lightweight architectures for focus-quality classification.

Both are depthwise-separable CNNs designed from scratch for this 3-class task
(Focused / Medium / Blur). They are validated in model_zoo.py alongside 13
pretrained backbones -- see PROJECT.md: these are NOT the headline
contribution of the paper, they are one entry among many in the backbone
comparison table.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv (per-channel) + pointwise conv (1x1), the MobileNet building block."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                    padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class FocusNetNano(nn.Module):
    """
    Minimal depthwise-separable CNN, ~0.3-0.5M params.
    5 blocks, stride-2 downsampling every block, no residual connections --
    optimized purely for smallest size / fastest inference.
    """

    def __init__(self, num_classes=3, width=44):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(w, w * 2, stride=2),        # 112 -> 56
            DepthwiseSeparableConv(w * 2, w * 4, stride=2),    # 56 -> 28
            DepthwiseSeparableConv(w * 4, w * 8, stride=2),    # 28 -> 14
            DepthwiseSeparableConv(w * 8, w * 16, stride=2),   # 14 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(w * 16, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class FocusNetMicroBlock(nn.Module):
    """Depthwise-separable block with a lightweight residual connection when shapes match."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch, out_ch, stride=stride)
        self.use_residual = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class FocusNetMicro(nn.Module):
    """
    Deeper depthwise-separable CNN, ~1-1.5M params.
    7 blocks with residual connections where shape allows -- targets accuracy
    closer to EfficientNet-B0 while staying far smaller.
    """

    def __init__(self, num_classes=3, width=44):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            FocusNetMicroBlock(w, w * 2, stride=2),         # 112 -> 56
            FocusNetMicroBlock(w * 2, w * 2, stride=1),     # residual
            FocusNetMicroBlock(w * 2, w * 4, stride=2),     # 56 -> 28
            FocusNetMicroBlock(w * 4, w * 4, stride=1),     # residual
            FocusNetMicroBlock(w * 4, w * 8, stride=2),     # 28 -> 14
            FocusNetMicroBlock(w * 8, w * 8, stride=1),     # residual
            FocusNetMicroBlock(w * 8, w * 16, stride=2),    # 14 -> 7
            FocusNetMicroBlock(w * 16, w * 16, stride=1),   # residual
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(w * 16, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class FocusNetSmall(nn.Module):
    """
    Deepest depthwise-separable CNN in the FocusNet family, ~2-3M params.
    Same 4-stage structure as Micro, but each stage has TWO residual blocks
    (not one) after the downsampling block, and a wider channel multiplier.
    Targets the gap between FocusNet-Micro (1M) and the smallest pretrained
    backbones (MobileNetV3-Small 2.5M, EfficientNet-B0 5M) on the params-vs-
    macro-F1 trade-off curve -- see PROJECT.md decision log.
    """

    def __init__(self, num_classes=3, width=56):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            FocusNetMicroBlock(w, w * 2, stride=2),         # 112 -> 56
            FocusNetMicroBlock(w * 2, w * 2, stride=1),     # residual
            FocusNetMicroBlock(w * 2, w * 2, stride=1),     # residual
            FocusNetMicroBlock(w * 2, w * 4, stride=2),     # 56 -> 28
            FocusNetMicroBlock(w * 4, w * 4, stride=1),     # residual
            FocusNetMicroBlock(w * 4, w * 4, stride=1),     # residual
            FocusNetMicroBlock(w * 4, w * 8, stride=2),     # 28 -> 14
            FocusNetMicroBlock(w * 8, w * 8, stride=1),     # residual
            FocusNetMicroBlock(w * 8, w * 8, stride=1),     # residual
            FocusNetMicroBlock(w * 8, w * 16, stride=2),    # 14 -> 7
            FocusNetMicroBlock(w * 16, w * 16, stride=1),   # residual
            FocusNetMicroBlock(w * 16, w * 16, stride=1),   # residual
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(w * 16, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    for cls in [FocusNetNano, FocusNetMicro, FocusNetSmall]:
        m = cls(num_classes=3)
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        print(f"{cls.__name__}: {n_params:,} params, output shape {tuple(out.shape)}")
