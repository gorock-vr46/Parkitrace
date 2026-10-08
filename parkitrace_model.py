"""ParkiTrace model, preprocessing and Grad-CAM utilities."""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

CLASSES = ["healthy", "parkinson"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ParkiTraceNet(nn.Module):
    """EfficientNetV2-S + ViT-B/16 feature fusion.

    The backbones can be frozen during training, which makes the project
    much more practical on CPU-only laptops.
    """

    def __init__(self, pretrained=True, freeze_backbones=False):
        super().__init__()
        eff_weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        vit_weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        eff = models.efficientnet_v2_s(weights=eff_weights)
        self.cnn = eff.features
        self.vit = models.vit_b_16(weights=vit_weights)
        self.vit.heads = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(1280 + 768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, 2),
        )
        if freeze_backbones:
            for module in (self.cnn, self.vit):
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x, return_fmap=False):
        fmap = self.cnn(x)
        local = self.pool(fmap).flatten(1)
        global_features = self.vit(x)
        out = self.head(torch.cat([local, global_features], dim=1))
        return (out, fmap) if return_fmap else out


def preprocess(bgr):
    if bgr is None or bgr.size == 0:
        raise ValueError("The image is empty.")
    bgr = cv2.medianBlur(bgr, 3)
    rgb = cv2.cvtColor(
        cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, rgb


def gradcam(model, x, cls):
    """Grad-CAM for the EfficientNet feature branch with optimized memory allocation."""
    was_training = model.training
    model.eval()

    # Freeze model parameters so autograd does not store unwanted layer gradients in RAM
    for param in model.parameters():
        param.requires_grad = False

    x_in = x.detach().clone()
    x_in.requires_grad = True

    with torch.enable_grad():
        out, fmap = model(x_in, return_fmap=True)
        score = out[0, cls]
        grads = torch.autograd.grad(score, fmap, retain_graph=False, create_graph=False)[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * fmap).sum(dim=1))[0].detach().cpu().numpy()

    # Explicitly clear graph tensors
    del fmap, grads, out, x_in
    model.zero_grad(set_to_none=True)

    cam -= cam.min()
    max_value = cam.max()
    if max_value > 0:
        cam /= max_value
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

    if was_training:
        model.train()
    return cam


def overlay(rgb, cam):
    heat = cv2.cvtColor(
        cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    return cv2.addWeighted(rgb, 0.55, heat, 0.45, 0)
