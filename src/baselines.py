# -*- coding: utf-8 -*-
"""src/baselines.py — Swin-T and InceptionV3 via timm."""
import timm


def build_swin_tiny(pretrained: bool = False):
    return timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=1,
    )


def build_inception_v3(pretrained: bool = False):
    model = timm.create_model(
        "inception_v3",
        pretrained=pretrained,
        num_classes=1,
    )
    # Disable auxiliary logits if the timm model exposes them
    if hasattr(model, "aux_logits"):
        model.aux_logits = False
    if hasattr(model, "AuxLogits"):
        model.AuxLogits = None
    return model
