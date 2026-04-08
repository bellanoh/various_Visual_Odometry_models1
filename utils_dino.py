# -*- coding: utf-8 -*-


import os
import math
import torch
import numpy as np
import random
from typing import Optional
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F


# =========================================
# Constants
# =========================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



# =========================================
# Checkpoint Save / Load
# =========================================
def save_checkpoint_full(path, epoch, model, optimizer, best_val_loss,
                         scaler: Optional[torch.cuda.amp.GradScaler] = None,
                         config: Optional[dict] = None):
    """Save a full training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config or {},
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)
    print(f"[Checkpoint] saved: {path}")


def load_checkpoint_full(path, model, optimizer=None, scaler=None, map_location=None):
    """Load a full training checkpoint."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optim" in state:
        optimizer.load_state_dict(state["optim"])
    if scaler is not None and "scaler" in state and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    best = state.get("best_val_loss", math.inf)
    epoch = state.get("epoch", 0)
    return epoch, best, state


# =========================================
# Label Normalization / Denormalization
# =========================================
def denormalize_labels(arr, stds, means):
    """
    x = z * std + mean
    arr: np.ndarray or torch.Tensor (..., D)
    stds/means: (D,)
    returns np.ndarray (float32)
    """

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    stds = np.asarray(stds).reshape((1,) * (arr.ndim - 1) + (arr.shape[-1],))
    means = np.asarray(means).reshape((1,) * (arr.ndim - 1) + (arr.shape[-1],))
    return (arr * stds + means).astype(np.float32)


# =========================================
# Sequence-Consistent Transform
# =========================================
class SeqTransform(nn.Module):
    def __init__(self, image_size=224):
        super().__init__()
        self.image_size = image_size

    def forward(self, seq):
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq).float()   # [2, H, W, 1]

        # Grayscale → 3채널
        if seq.shape[-1] == 1:
            seq = seq.repeat(1, 1, 1, 3)          # [2, H, W, 3]

        # [2, H, W, 3] → [2, 3, H, W]
        seq = seq.permute(0, 3, 1, 2)

        # Resize만 수행 (Normalize는 processor가 함)
        seq = F.interpolate(
            seq,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )  # [2, 3, 224, 224]

        # 0~255 범위라면 0~1로 스케일링 (중요!)
        if seq.max() > 1.0:
            seq = seq / 255.0

        return seq



