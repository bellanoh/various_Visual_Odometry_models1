# -*- coding: utf-8 -*-


import os
import math
import torch
import numpy as np
import random
from typing import Optional
from torchvision.transforms import InterpolationMode
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import numpy as np


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
    """
    Sequence-consistent preprocessing for ultrasound image sequences.

    Pipeline:
        Resize -> Normalize (ImageNet)

    Input :
        tensor (T, 3, H, W) in [0, 1]
    Output:
        tensor (T, 3, S, S) normalized by ImageNet statistics
    """

    def __init__(self, image_size=32, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.image_size = image_size
        self.mean = mean  # RGB mean (list of 3)
        self.std = std  # RGB std (list of 3)
        # Grayscale용 mean/std 계산: RGB 평균 사용 (일반적인 grayscale 변환)
        self.mean_gray = [sum(self.mean) / 3]  # e.g., ≈0.449
        self.std_gray = [sum(self.std) / 3]  # e.g., ≈0.226

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if isinstance(seq, np.ndarray):
            seq = torch.from_numpy(seq).float()  # numpy → tensor

        # 입력 형태 확인 & 4D로 맞추기: 단일 이미지 처리 (2D or 3D → (1, C, H, W))
        if seq.ndim == 2:  # grayscale 단일 이미지 (H, W)
            seq = seq.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) – batch와 channel 1 추가
        elif seq.ndim == 3:  # (H, W, C) – C=1 (grayscale) 또는 C=3 (RGB)
            if seq.shape[2] == 3:  # RGB인 경우 grayscale로 변환 (사용자 의도: 1채널 유지)
                seq = torch.mean(seq, dim=2, keepdim=True)  # (H, W, 1)
            seq = seq.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W) 또는 (1, 3, H, W) but forced to 1
        elif seq.ndim == 4:  # 이미 시퀀스 (T, H, W, C)
            if seq.shape[3] == 3:  # RGB인 경우 grayscale로 변환
                seq = torch.mean(seq, dim=3, keepdim=True)  # (T, H, W, 1)
            seq = seq.permute(0, 3, 1, 2)  # (T, 1, H, W)
        else:
            raise ValueError(f"Unexpected seq shape: {seq.shape}")

        # Resize all frames (treat T as batch dimension)
        seq = F.interpolate(
            seq, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )  # (T, C, S, S) – C=1 유지

        # 채널 수에 따라 적절한 mean/std 선택 후 Normalize
        if seq.shape[1] == 1:  # 1채널 (grayscale)
            norm_mean = self.mean_gray
            norm_std = self.std_gray
        else:  # 3채널 (fallback, but won't reach if forced to 1)
            norm_mean = self.mean
            norm_std = self.std

        # Normalize each frame
        seq = torch.stack(
            [TF.normalize(seq[i], mean=norm_mean, std=norm_std) for i in range(seq.shape[0])],
            dim=0
        )
        return seq


