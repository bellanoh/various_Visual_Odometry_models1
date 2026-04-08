
# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from utils_dino import SeqTransform, save_checkpoint_full, denormalize_labels
from torch.utils.data import default_collate
from torch.nn.utils.rnn import pad_sequence
import torch
import random
from tqdm import tqdm

# =========================================
# Load normalization parameters
# =========================================
stds, means = np.load(os.path.join(DATA_DIR, "norm_params.npy"))

# =========================================
# Test Dataset & DataLoader
# =========================================
test_tf = SeqTransform(image_size=IMAGE_SIZE)

test_dataset = DinoVODataset(
    pairs_file=os.path.join(DATA_DIR, "test_pairs.npy"),
    labels_file=os.path.join(DATA_DIR, "test_pair_labels.npy"),
    keypoints_file=os.path.join(DATA_DIR, "test_keypoints.pkl"),
    transform=test_tf,
    label_stds=stds,
    label_means=means
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=custom_collate,
    drop_last=False          # ← test에서는 drop_last=False가 더 좋음
)

# =========================================
# Load Model
# =========================================
model = DinoVOModel(model_name="facebook/dinov2-small").to(DEVICE)

state_path = os.path.join(RESULT_DIR, "best_model.pth")
state_dict = torch.load(state_path, map_location=DEVICE)

model.load_state_dict(state_dict)

model.eval()

preds_z = []
gts_z = []

NOISE_PROB = 0.3
NOISE_STD = 0.08  # 노이즈 강도 (필요시 0.10 ~ 0.12로 조정 가능)

with torch.no_grad():
    for batch_idx, (anchor, positive, ys, kp_anchor, kp_positive) in enumerate(
            tqdm(test_loader, desc="[Test + Noise]")
    ):

        # GPU로 이동
        anchor = anchor.to(DEVICE, non_blocking=True)
        positive = positive.to(DEVICE, non_blocking=True)
        kp_anchor = kp_anchor.to(DEVICE, non_blocking=True)
        kp_positive = kp_positive.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=True)

        # ====================== Gaussian Noise 추가 ======================
        if random.random() < NOISE_PROB:
            # anchor와 positive 이미지에만 노이즈 적용
            noise_anchor = torch.randn_like(anchor) * NOISE_STD
            noise_positive = torch.randn_like(positive) * NOISE_STD

            anchor = torch.clamp(anchor + noise_anchor, 0.0, 1.0)
            positive = torch.clamp(positive + noise_positive, 0.0, 1.0)

            if (batch_idx + 1) % 30 == 0:
                print(f"Batch {batch_idx + 1:4d}: Gaussian Noise added to anchor & positive (std={NOISE_STD})")

        # ====================== Forward ======================
        outs = model(anchor, positive, kp_anchor, kp_positive)

        # tuple 반환 처리
        if isinstance(outs, tuple):
            outs = outs[0]

        preds_z.append(outs.cpu().numpy())
        gts_z.append(ys[:, 0:3].cpu().numpy())  # pose label 3개만 사용

# Concatenate
preds_z = np.concatenate(preds_z, axis=0)
gts_z = np.concatenate(gts_z, axis=0)
# =========================================
# Denormalize
# =========================================
preds_dn = denormalize_labels(preds_z, stds, means)
gts_dn   = denormalize_labels(gts_z,   stds, means)


