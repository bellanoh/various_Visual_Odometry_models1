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

    # if isinstance(arr, torch.Tensor):
    #     arr = arr.detach().cpu().numpy()
    # arr = np.asarray(arr)
    # stds = np.asarray(stds).reshape((1,) + stds.shape)
    # means = np.asarray(means).reshape((1,) + means.shape)
    # return (arr * stds + means).astype(np.float32)


# =========================================
# Sequence-Consistent Transform
# =========================================
class SeqTransform(nn.Module):
    """
    Sequence-consistent preprocessing for ultrasound image sequences.

    Pipeline:
        Resize -> Normalize (ImageNet)

    Input :
        tensor (T, 3, H, W) in [0, 1]
    Output:
        tensor (T, 3, S, S) normalized by ImageNet statistics
    """
    def __init__(self, image_size=224,
                 mean=IMAGENET_MEAN,
                 std=IMAGENET_STD):
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.std = std

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # Resize all frames (treat T as batch dimension)
        seq = torch.nn.functional.interpolate(
            seq,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )  # (T, 3, S, S)

        # Normalize each frame using ImageNet statistics
        seq = torch.stack(
            [TF.normalize(seq[i], mean=self.mean, std=self.std)
             for i in range(seq.shape[0])],
            dim=0
        )
        return seq


