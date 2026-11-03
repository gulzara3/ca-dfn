"""
Base-model factory for the five architecturally diverse backbones.

Each backbone is ImageNet-1K pretrained and fine-tuned with a differential
learning rate: the backbone at LR * 0.1 and the classification head at LR.
"""
import torch.nn as nn
import torchvision.models as models
import timm

from .config import CFG, MODEL_NAMES, DISPLAY, MODEL_TYPE


def build_model(name, nc=CFG.NUM_CLASSES, pretrained=True):
    """Return (model, param_groups) for the named backbone.

    param_groups sets backbone LR to CFG.LR * 0.1 and head LR to CFG.LR.
    """
    if name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, nc)
        bb = [p for n, p in m.named_parameters() if "fc" not in n]
        hd = list(m.fc.parameters())
    elif name == "efficientnet_b4":
        m = models.efficientnet_b4(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
        bb = [p for n, p in m.named_parameters() if "classifier" not in n]
        hd = list(m.classifier.parameters())
    elif name == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, nc)
        bb = [p for n, p in m.named_parameters() if "classifier" not in n]
        hd = list(m.classifier.parameters())
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, nc)
        bb = [p for n, p in m.named_parameters() if "classifier" not in n]
        hd = list(m.classifier.parameters())
    elif name == "swin_small":
        m = timm.create_model("swin_small_patch4_window7_224",
                              pretrained=pretrained, num_classes=nc)
        bb = [p for n, p in m.named_parameters() if "head" not in n]
        hd = [p for n, p in m.named_parameters() if "head" in n]
    else:
        raise ValueError(f"Unknown model: {name}")

    param_groups = [
        {"params": bb, "lr": CFG.LR * 0.1},
        {"params": hd, "lr": CFG.LR},
    ]
    return m.to(CFG.DEVICE), param_groups


def summarise_models():
    """Print a parameter-count table for all five backbones."""
    import torch
    print(f"{'Model':<22s} {'Type':<12s} {'Params':>8s}")
    print("-" * 44)
    for nm in MODEL_NAMES:
        tmp, _ = build_model(nm, pretrained=False)
        n_params = sum(p.numel() for p in tmp.parameters()) / 1e6
        print(f"{DISPLAY[nm]:<22s} {MODEL_TYPE[nm]:<12s} {n_params:>7.1f}M")
        del tmp
    torch.cuda.empty_cache()
