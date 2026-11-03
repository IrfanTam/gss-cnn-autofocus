"""
Single entry point for all 15 backbone architectures evaluated in the
hybrid GSS-CNN autofocus pipeline. See PROJECT.md for the full rationale
and the fixed list -- do not add/remove entries here without updating
PROJECT.md first.

Usage:
    from model_zoo import get_model, MODEL_REGISTRY
    model = get_model("efficientnet_b0", num_classes=3)
"""

import torchvision.models as tvm
import torch.nn as nn
import timm

from focusnet import FocusNetNano, FocusNetMicro, FocusNetSmall

# name -> timm model string, for the 12 backbones available via timm
TIMM_MODELS = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "mobilenetv3_small": "mobilenetv3_small_100",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "regnety_400mf": "regnety_004",
    "convnext_tiny": "convnext_tiny",
    "vit_tiny": "vit_tiny_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}

# Custom backbones, built from scratch (see focusnet.py)
CUSTOM_MODELS = {
    "focusnet_nano": FocusNetNano,
    "focusnet_micro": FocusNetMicro,
    "focusnet_small": FocusNetSmall,
}

# Full registry used to drive the training loop -- 15 entries total.
# "source" controls which branch of get_model() builds it.
# YOLO26n is NOT included here: it is trained via the separate Ultralytics
# classification pipeline (see train_yolo26n.py), not this generic loop.
MODEL_REGISTRY = {
    **{name: dict(source="timm") for name in TIMM_MODELS},
    "shufflenet_v2": dict(source="torchvision"),
    **{name: dict(source="custom") for name in CUSTOM_MODELS},
}


def _build_shufflenet_v2(num_classes, pretrained=True):
    weights = tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.shufflenet_v2_x1_0(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(name, num_classes=3, pretrained=True):
    """Build a model by registry name, adapted to num_classes outputs."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")

    source = MODEL_REGISTRY[name]["source"]

    if source == "timm":
        return timm.create_model(TIMM_MODELS[name], pretrained=pretrained, num_classes=num_classes)

    if source == "torchvision":
        if name == "shufflenet_v2":
            return _build_shufflenet_v2(num_classes, pretrained=pretrained)
        raise ValueError(f"No torchvision builder for '{name}'")

    if source == "custom":
        return CUSTOM_MODELS[name](num_classes=num_classes)

    raise ValueError(f"Unknown source '{source}' for model '{name}'")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import torch

    x = torch.randn(2, 3, 224, 224)
    print(f"{'Model':22s} {'Params':>12s}  {'Output shape'}")
    print("-" * 50)
    for name in MODEL_REGISTRY:
        model = get_model(name, num_classes=3, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"{name:22s} {count_params(model):>12,}  {tuple(out.shape)}")
