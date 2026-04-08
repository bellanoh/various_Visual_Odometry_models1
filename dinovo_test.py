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


def custom_collate(batch):
    """
    batch: list of (anchor, positive, label, kp_anchor, kp_positive)
    - kp_anchor, kp_positive: Dataset에서 이미 torch.Tensor([N, 2]) 형태로 변환된 상태
    """
    # 1. 이미지와 label 처리
    anchors, positives, labels = default_collate([
        (item[0], item[1], item[2]) for item in batch
    ])

    # 2. keypoints 처리 (이미 tensor지만, 안전하게 처리)
    kp_anchors = []
    kp_positives = []

    for item in batch:
        kp_a = item[3]  # torch.Tensor [N, 2] 또는 numpy
        kp_p = item[4]

        # tensor가 아니면 변환 (numpy나 list일 경우 대비)
        if not isinstance(kp_a, torch.Tensor):
            kp_a = torch.tensor(kp_a, dtype=torch.float32)
        if not isinstance(kp_p, torch.Tensor):
            kp_p = torch.tensor(kp_p, dtype=torch.float32)

        kp_anchors.append(kp_a)
        kp_positives.append(kp_p)

    # 3. 가변 길이 N → Padding 처리
    kp_anchor_batch = pad_sequence(kp_anchors, batch_first=True, padding_value=0.0)  # [B, max_N, 2]
    kp_positive_batch = pad_sequence(kp_positives, batch_first=True, padding_value=0.0)

    return anchors, positives, labels, kp_anchor_batch, kp_positive_batch
# ====================== 2. Feature Descriptor (DINOv2 + FinerCNN + Keypoint queries) ======================
class DinoV2WithFiner(nn.Module):
    """
    DINO-VO 스타일 Feature Descriptor (dinov2-small 버전)
    - Keypoints를 query로 사용하여 DINOv2 + FinerCNN에서 feature를 추출
    """
    def __init__(self, model_name="facebook/dinov2-small", embed_dim=384):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.dinov2 = AutoModel.from_pretrained(model_name)
        self.embed_dim = embed_dim

        # Lightweight CNN Encoder (FinerCNN) - embed_dim에 맞춰 조정
        self.finer_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3, stride=2, padding=1),   # 384 유지 (fusion을 위해)
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
        )

        # Keypoint Query Projection
        self.keypoint_proj = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim)          # 384로 출력
        )

        # Final fusion projection (DINOv2 384 + FinerCNN 384 + Keypoint 384)
        self.fusion_proj = nn.Linear(embed_dim * 3, embed_dim)   # 384*3 → 384

    def forward(self, images, keypoints):
        B, N_padded, _ = keypoints.shape
        device = images.device

        # Padding Mask
        is_padding = (keypoints.abs() < 1e-5).all(dim=-1)
        mask = ~is_padding

        # ==================== Processor 호출 (가장 중요) ====================
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_rescale=True,  # 0~1 범위라면 True
            do_normalize=True  # ImageNet mean/std 적용
        ).to(device)

        dinov2_out = self.dinov2(**inputs)
        patch_tokens = dinov2_out.last_hidden_state[:, 1:, :]

        # FinerCNN은 normalize된 이미지를 사용하는 게 좋음
        # (필요하면 images = TF.normalize(images, mean=..., std=...) 추가)

        cnn_feat = self.finer_cnn(images)
        cnn_feat = cnn_feat.flatten(2).transpose(1, 2)

        # 2. Keypoint Query Embedding
        kp_query = self.keypoint_proj(keypoints)  # [B, max_N, 384]

        # 3. Grid Sampling
        grid = keypoints * 2.0 - 1.0  # [-1, 1] 범위로 변환
        grid = grid.unsqueeze(2)  # [B, max_N, 1, 2]

        sampled_dino = F.grid_sample(
            patch_tokens.transpose(1, 2).unsqueeze(3),  # [B, 384, 196, 1]
            grid, mode='bilinear', align_corners=False
        ).squeeze(3).transpose(1, 2)  # [B, max_N, 384]

        sampled_cnn = F.grid_sample(
            cnn_feat.transpose(1, 2).unsqueeze(3),
            grid, mode='bilinear', align_corners=False
        ).squeeze(3).transpose(1, 2)  # [B, max_N, 384]

        # 4. Fusion
        fused = torch.cat([sampled_dino, sampled_cnn, kp_query], dim=-1)  # [B, max_N, 384*3]
        fused = self.fusion_proj(fused)  # [B, max_N, 384]

        # ==================== Mask 적용 ====================
        # 패딩된 위치의 feature를 완전히 0으로 만듦
        fused = fused * mask.unsqueeze(-1).float()  # [B, max_N, 384]

        return fused


# ====================== 2. Feature Transformer (Self + Cross Attention) ======================
class FeatureTransformer(nn.Module):
    """
    Self + Cross Attention with proper handling of different keypoint counts
    """
    def __init__(self, embed_dim=384, num_heads=6, num_layers=3, max_keypoints=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_keypoints = max_keypoints

        # Learnable Positional Embedding
        self.pos_embed_anchor = nn.Parameter(torch.zeros(1, max_keypoints, embed_dim))
        self.pos_embed_positive = nn.Parameter(torch.zeros(1, max_keypoints, embed_dim))

        # Self Attention Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            norm_first=True,
            batch_first=True,
            dropout=0.1
        )
        self.self_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross Attention
        self.cross_a2p = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.cross_p2a = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)

        # LayerNorm
        self.norm_anchor = nn.LayerNorm(embed_dim)
        self.norm_positive = nn.LayerNorm(embed_dim)

    def forward(self, anchor_feat, positive_feat, kp_anchor=None, kp_positive=None):
        """
        anchor_feat, positive_feat: [B, N, embed_dim]   ← N이 anchor와 positive마다 다를 수 있음
        """
        B, N_a, D = anchor_feat.shape   # anchor의 실제 키포인트 수
        _, N_p, _ = positive_feat.shape # positive의 실제 키포인트 수

        # ==================== Positional Embedding (각각 다른 N에 맞춤) ====================
        pos_anchor = self.pos_embed_anchor[:, :N_a, :]      # [1, N_a, D]
        pos_positive = self.pos_embed_positive[:, :N_p, :]  # [1, N_p, D]

        anchor_feat = anchor_feat + pos_anchor
        positive_feat = positive_feat + pos_positive

        # ==================== Self Attention ====================
        anchor_self = self.self_transformer(anchor_feat)
        positive_self = self.self_transformer(positive_feat)

        # ==================== Cross Attention ====================
        # Anchor queries Positive
        anchor_cross, _ = self.cross_a2p(
            query=anchor_self,
            key=positive_self,
            value=positive_self
        )

        # Positive queries Anchor
        positive_cross, _ = self.cross_p2a(
            query=positive_self,
            key=anchor_self,
            value=anchor_self
        )

        # Residual + LayerNorm
        anchor_matched = self.norm_anchor(anchor_self + anchor_cross)
        positive_matched = self.norm_positive(positive_self + positive_cross)

        return anchor_matched, positive_matched


# ====================== 3. Matching Layer + 3-DoF Pose Regression Head ======================
class DinoVOModel(nn.Module):
    """
    DINO-VO 스타일 전체 모델
    - Confidence-weighted Global Pooling 적용
    """
    def __init__(self, model_name="facebook/dinov2-small", embed_dim=384, num_keypoints=512):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.embed_dim = embed_dim

        # 1. Feature Descriptor
        self.descriptor = DinoV2WithFiner(model_name=model_name, embed_dim=embed_dim)

        # 2. Feature Transformer (Self + Cross Attention)
        self.transformer = FeatureTransformer(
            embed_dim=embed_dim,
            num_heads=6,
            num_layers=3,
            max_keypoints=num_keypoints
        )

        # 3. Confidence MLP (각 키포인트의 신뢰도 예측)
        self.conf_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()                    # 0~1 사이 confidence
        )

        # 4. Pose Regression Head
        self.pose_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 3)         # tx, ty, tz
        )

    def forward(self, anchor_img, positive_img, kp_anchor, kp_positive):
        """
        anchor_img, positive_img: [B, 3, 224, 224]
        kp_anchor, kp_positive  : [B, max_N, 2]
        return: pose [B, 3]
        """
        # 1. Feature Descriptor
        anchor_feat = self.descriptor(anchor_img, kp_anchor)      # [B, max_N, embed_dim]
        positive_feat = self.descriptor(positive_img, kp_positive)

        # 2. Feature Matching (Self + Cross Attention)
        anchor_matched, positive_matched = self.transformer(
            anchor_feat, positive_feat, kp_anchor, kp_positive
        )

        # ==================== Confidence-weighted Pooling ====================
        # Confidence score 계산
        conf_anchor = self.conf_mlp(anchor_matched)      # [B, max_N, 1]
        conf_positive = self.conf_mlp(positive_matched)

        # Padding Mask (0으로 패딩된 키포인트 제외)
        mask_a = (kp_anchor.abs().sum(dim=-1) > 1e-5).unsqueeze(-1).float()  # [B, max_N, 1]
        mask_p = (kp_positive.abs().sum(dim=-1) > 1e-5).unsqueeze(-1).float()

        # Confidence * Mask 결합 (패딩된 부분은 confidence=0으로 강제)
        weight_anchor = conf_anchor * mask_a
        weight_positive = conf_positive * mask_p

        # Weighted Average Pooling
        # sum(weight * feature) / sum(weight)
        anchor_global = (weight_anchor * anchor_matched).sum(dim=1) / \
                        (weight_anchor.sum(dim=1) + 1e-8)   # [B, embed_dim]

        positive_global = (weight_positive * positive_matched).sum(dim=1) / \
                          (weight_positive.sum(dim=1) + 1e-8)

        # 3. Pose Regression
        combined = torch.cat([anchor_global, positive_global], dim=1)   # [B, 768]
        pose_3dof = self.pose_head(combined)                            # [B, 3]

        return pose_3dof
class DinoVODataset(torch.utils.data.Dataset):
    def __init__(self, pairs_file, labels_file, keypoints_file=None,
                 transform=None, label_stds=None, label_means=None):
        self.pairs = np.load(pairs_file)           # [N, 2, H, W, 3]
        self.labels = np.load(labels_file)         # [N, num_labels]
        self.transform = transform
        self.label_stds = label_stds
        self.label_means = label_means

        # keypoints 파일 로드 (pickle)
        if keypoints_file is not None:
            import pickle
            with open(keypoints_file, "rb") as f:
                self.keypoints = pickle.load(f)    # list of dict
        else:
            self.keypoints = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]  # [2, H, W, 1] 또는 [2, H, W, 3]

        # Transform에 pair 전체를 넘김 → [2, 3, 224, 224] 형태로 나옴
        if self.transform:
            pair_transformed = self.transform(pair)  # [2, 3, 224, 224]
            anchor = pair_transformed[0]  # [3, 224, 224]
            positive = pair_transformed[1]
        else:
            # transform이 없을 경우 fallback
            anchor = torch.from_numpy(pair[0]).permute(2, 0, 1).float()
            positive = torch.from_numpy(pair[1]).permute(2, 0, 1).float()

        label = torch.tensor(self.labels[idx]).float()

        if self.label_stds is not None and self.label_means is not None:
            label = (label - self.label_means) / self.label_stds

        # Keypoints
        if self.keypoints is not None:
            kp_dict = self.keypoints[idx]
            kp_anchor = torch.from_numpy(kp_dict['anchor']).float()
            kp_positive = torch.from_numpy(kp_dict['positive']).float()
        else:
            kp_anchor = kp_positive = torch.zeros((512, 2), dtype=torch.float32)

        return anchor, positive, label, kp_anchor, kp_positive

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
model.eval()                    # ← 매우 중요!

print(f"Model loaded from {state_path}")

# =========================================
# Inference
# =========================================
preds_z = []
gts_z = []

with torch.no_grad():
    for anchor, positive, ys, kp_anchor, kp_positive in tqdm(test_loader, desc="[Test]"):
        anchor = anchor.to(DEVICE, non_blocking=True)
        positive = positive.to(DEVICE, non_blocking=True)
        kp_anchor = kp_anchor.to(DEVICE, non_blocking=True)
        kp_positive = kp_positive.to(DEVICE, non_blocking=True)
        ys = ys.to(DEVICE, non_blocking=True)

        # Forward
        outs = model(anchor, positive, kp_anchor, kp_positive)   # [B, 3]

        # 혹시 tuple로 나오면 첫 번째 요소만 사용
        if isinstance(outs, tuple):
            outs = outs[0]

        preds_z.append(outs.cpu().numpy())
        gts_z.append(ys[:, 0:3].cpu().numpy())   # labels[:,0:3]만 사용

# Concatenate
preds_z = np.concatenate(preds_z, axis=0)
gts_z = np.concatenate(gts_z, axis=0)


# =========================================
# Denormalize
# =========================================
preds_dn = denormalize_labels(preds_z, stds, means)
gts_dn   = denormalize_labels(gts_z,   stds, means)
