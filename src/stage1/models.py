"""Stage 1 model architectures.

Phase A: DINOv2 ViT-B/14 frozen backbone + MLP head with 3 prediction branches
(building_type, year, num_floors). Per spec section 3.2.3 the per-building
aggregation is mean-pool over backbone features of valid images.

Phase C: ResNet-50 full fine-tune (`ResNet50FT`). Unlike the frozen path it
packs only the valid images through the backbone: the dataset zero-pads every
building to MAX_IMAGES slots, and in train mode those all-black images would
corrupt the BatchNorm batch statistics (~40% of slots are padding). Packing
also saves the wasted forward/backward compute.

Swin-T full fine-tune remains a placeholder.
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


class DINOv2FrozenEnergy(nn.Module):
    """Aligned M2: frozen DINOv2 backbone + trunk + single 7-class energy head.

    Architecturally identical to `DINOv2FrozenMLP` (same frozen backbone, same
    trunk) so that M2 (image -> energy label, end-to-end) vs M3 (image -> 3
    attributes -> TABULA -> LightGBM) isolates "end-to-end vs decomposed" rather
    than a head/architecture change. Only the output head differs: one 7-way
    Energieklasse head instead of the three attribute heads.
    """

    def __init__(self, num_energy_classes: int = 7, dropout: float = 0.3):
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
        self.head_energy = nn.Linear(hidden, num_energy_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        B, K, C, H, W = images.shape
        flat = images.view(B * K, C, H, W)
        with torch.no_grad():
            feats = self.backbone(flat)
        feats = feats.view(B, K, -1)

        mask = valid_mask.unsqueeze(-1).float()
        n_valid = mask.sum(dim=1).clamp(min=1.0)
        pooled = (feats * mask).sum(dim=1) / n_valid

        trunk = self.trunk(pooled)
        return {"logits_energy": self.head_energy(trunk)}

    def trainable_parameters(self):
        return list(self.trunk.parameters()) + list(self.head_energy.parameters())


class ResNet50FT(nn.Module):
    """ResNet-50 full fine-tune + small MLP with 3 heads.

    Differences from the frozen DINOv2 path:
    - gradients flow through the backbone (no no_grad, no forced eval)
    - only valid images are run through the backbone (packing), so padded
      zero-images never enter the BatchNorm batch statistics
    - `param_groups()` provides discriminative lr (pretrained backbone vs
      randomly initialised trunk/heads) with norm/bias excluded from decay
    """

    RESNET_FEAT_DIM = 2048

    def __init__(self, num_type_classes: int = NUM_TYPE_CLASSES, dropout: float = 0.3):
        super().__init__()

        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)

        feat_dim = self.RESNET_FEAT_DIM
        hidden = 256
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_type = nn.Linear(hidden, num_type_classes)
        self.head_year = nn.Linear(hidden, 1)
        self.head_floors = nn.Linear(hidden, 1)

        self._bn_eval = False

    def set_bn_eval(self, enabled: bool = True):
        """Freeze backbone BN running stats (fallback for batch < 8 buildings).

        gamma/beta keep training; only the running mean/var stay at the
        pretrained ImageNet values."""
        self._bn_eval = enabled
        if enabled:
            self._apply_bn_eval()
        return self

    def _apply_bn_eval(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._bn_eval:
            self._apply_bn_eval()
        return self

    def forward(self, images: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        """
        images: [B, K, 3, H, W] (zero-padded to K slots)
        valid_mask: [B, K] bool — True for real images, False for padding
        """
        B, K, C, H, W = images.shape

        # pack: only real images go through the backbone
        flat_mask = valid_mask.reshape(B * K)
        packed = images.reshape(B * K, C, H, W)[flat_mask]
        packed_feats = self.backbone(packed)

        feats = packed_feats.new_zeros(B * K, packed_feats.shape[-1])
        feats[flat_mask] = packed_feats
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
        return list(self.parameters())

    def param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        """4 groups: {backbone, head} x {decay, no_decay(norm params + bias)}."""
        head_modules = [self.trunk, self.head_type, self.head_year, self.head_floors]
        head_param_ids = {id(p) for m in head_modules for p in m.parameters()}

        groups = {
            "backbone_decay": [], "backbone_no_decay": [],
            "head_decay": [], "head_no_decay": [],
        }
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            part = "head" if id(p) in head_param_ids else "backbone"
            # 1-d params = norm gamma/beta and biases
            decay = "no_decay" if p.ndim <= 1 else "decay"
            groups[f"{part}_{decay}"].append(p)

        return [
            {"params": groups["backbone_decay"], "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": groups["backbone_no_decay"], "lr": backbone_lr, "weight_decay": 0.0},
            {"params": groups["head_decay"], "lr": head_lr, "weight_decay": weight_decay},
            {"params": groups["head_no_decay"], "lr": head_lr, "weight_decay": 0.0},
        ]


class ResNet50Energy(nn.Module):
    """Aligned M2 for ResNet-50: full fine-tune backbone + trunk + single 7-class
    energy head. Mirrors `ResNet50FT` (packed forward over valid images, BN-safe,
    discriminative lr via param_groups) so M2-ResNet50 (end-to-end -> energy) vs
    M3-ResNet50 (-> attributes -> TABULA -> LightGBM) isolates end-to-end vs
    decomposed for the same full-fine-tune ResNet paradigm.
    """

    RESNET_FEAT_DIM = 2048

    def __init__(self, num_energy_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        hidden = 256
        self.trunk = nn.Sequential(
            nn.Linear(self.RESNET_FEAT_DIM, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head_energy = nn.Linear(hidden, num_energy_classes)
        self._bn_eval = False

    def set_bn_eval(self, enabled: bool = True):
        self._bn_eval = enabled
        if enabled:
            self._apply_bn_eval()
        return self

    def _apply_bn_eval(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._bn_eval:
            self._apply_bn_eval()
        return self

    def forward(self, images: torch.Tensor, valid_mask: torch.Tensor) -> dict:
        B, K, C, H, W = images.shape
        flat_mask = valid_mask.reshape(B * K)
        packed = images.reshape(B * K, C, H, W)[flat_mask]
        packed_feats = self.backbone(packed)
        feats = packed_feats.new_zeros(B * K, packed_feats.shape[-1])
        feats[flat_mask] = packed_feats
        feats = feats.view(B, K, -1)
        mask = valid_mask.unsqueeze(-1).float()
        n_valid = mask.sum(dim=1).clamp(min=1.0)
        pooled = (feats * mask).sum(dim=1) / n_valid
        trunk = self.trunk(pooled)
        return {"logits_energy": self.head_energy(trunk)}

    def trainable_parameters(self):
        return list(self.parameters())

    def param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        head_modules = [self.trunk, self.head_energy]
        head_param_ids = {id(p) for m in head_modules for p in m.parameters()}
        groups = {"backbone_decay": [], "backbone_no_decay": [],
                  "head_decay": [], "head_no_decay": []}
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            part = "head" if id(p) in head_param_ids else "backbone"
            decay = "no_decay" if p.ndim <= 1 else "decay"
            groups[f"{part}_{decay}"].append(p)
        return [
            {"params": groups["backbone_decay"], "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": groups["backbone_no_decay"], "lr": backbone_lr, "weight_decay": 0.0},
            {"params": groups["head_decay"], "lr": head_lr, "weight_decay": weight_decay},
            {"params": groups["head_no_decay"], "lr": head_lr, "weight_decay": 0.0},
        ]


def build_model(name: str) -> nn.Module:
    if name == "dinov2_frozen":
        return DINOv2FrozenMLP()
    if name == "resnet50_ft":
        return ResNet50FT()
    if name == "dinov2_energy":
        return DINOv2FrozenEnergy()
    if name == "resnet50_energy":
        return ResNet50Energy()
    if name == "swin_t_ft":
        raise NotImplementedError("swin_t_ft reserved for a later phase")
    raise ValueError(f"unknown model name: {name}")
