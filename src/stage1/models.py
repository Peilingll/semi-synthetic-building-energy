"""Stage 1 model architectures.

Phase A: DINOv2 ViT-B/14 frozen backbone + MLP head with 3 prediction branches
(building_type, year, num_floors). Per spec section 3.2.3 the per-building
aggregation is mean-pool over backbone features of valid images.

Phase B placeholders: ResNet-50 / Swin-T full fine-tune (not implemented here).
"""

import logging

import timm
import torch
from torch import nn

logger = logging.getLogger(__name__)

NUM_TYPE_CLASSES = 4
DINOV2_FEAT_DIM = 768
DINOV2_MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"


class DINOv2FrozenMLP(nn.Module):
    """Frozen DINOv2 backbone + small MLP with 3 heads."""

    def __init__(self, num_type_classes: int = NUM_TYPE_CLASSES, dropout: float = 0.3):
        super().__init__()

        self.backbone = timm.create_model(
            DINOV2_MODEL_NAME,
            pretrained=True,
            num_classes=0,
            img_size=224,
            dynamic_img_size=True,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        feat_dim = DINOV2_FEAT_DIM
        hidden = 256
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_type = nn.Linear(hidden, num_type_classes)
        self.head_year = nn.Linear(hidden, 1)
        self.head_floors = nn.Linear(hidden, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        """
        images: [B, K, 3, H, W]
        valid_mask: [B, K] bool — True for real images, False for padding
        """
        B, K, C, H, W = images.shape
        flat = images.view(B * K, C, H, W)

        with torch.no_grad():
            feats = self.backbone(flat)
        feats = feats.view(B, K, -1)

        mask = valid_mask.unsqueeze(-1).float()
        n_valid = mask.sum(dim=1).clamp(min=1.0)
        pooled = (feats * mask).sum(dim=1) / n_valid

        trunk = self.trunk(pooled)
        return {
            "logits_type": self.head_type(trunk),
            "pred_year_norm": self.head_year(trunk).squeeze(-1),
            "pred_floors_norm": self.head_floors(trunk).squeeze(-1),
        }

    def trainable_parameters(self):
        return (
            list(self.trunk.parameters())
            + list(self.head_type.parameters())
            + list(self.head_year.parameters())
            + list(self.head_floors.parameters())
        )


def build_model(name: str) -> nn.Module:
    if name == "dinov2_frozen":
        return DINOv2FrozenMLP()
    if name in {"resnet50_ft", "swin_t_ft"}:
        raise NotImplementedError(f"{name} reserved for Phase B")
    raise ValueError(f"unknown model name: {name}")
