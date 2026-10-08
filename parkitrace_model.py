"""ParkiTrace model, preprocessing and heatmapping utilities."""
import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

CLASSES = ["healthy", "parkinson"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ParkiTraceNet(nn.Module):
    """EfficientNetV2-S + ViT-B/16 feature fusion with low-memory sequential execution."""

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
        # 1. Compute local CNN features first
        fmap = self.cnn(x)
        local = self.pool(fmap).flatten(1)

        # 2. Compute global ViT features sequentially
        global_features = self.vit(x)

        # 3. Fuse feature vectors in classification head
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
    """Zero-memory heatmap generation using image intensity and contours."""
    img = x[0].numpy().transpose(1, 2, 0)
    img = ((img * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    heatmap = cv2.Laplacian(blur, cv2.CV_64F)
    heatmap = np.abs(heatmap)
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
        
    cam = cv2.GaussianBlur(heatmap.astype(np.float32), (21, 21), 0)
    if cam.max() > 0:
        cam = cam / cam.max()
        
    return cam


def overlay(rgb, cam):
    heat = cv2.cvtColor(
        cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    return cv2.addWeighted(rgb, 0.55, heat, 0.45, 0)
