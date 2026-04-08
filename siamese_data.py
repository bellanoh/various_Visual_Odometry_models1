# -*- coding: utf-8 -*-


import os
import sys
import json

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import random


# =========================================
# Siamese Pair Generation
# =========================================
def make_pairs_from_dir_trainval(d, label_names, crop_prob=0.0):
    """
    연속된 바로 옆 프레임으로 pair 생성 + Relative Label 생성
    - label = positive_label - anchor_label (delta pose)
    """
    images = list_sorted_images(d)
    if len(images) < 2:
        return None, None

    labels = load_label_excel(d, EXCEL_NAME, label_names)   # [num_frames, num_labels]

    pairs = []
    pair_labels = []        # ← 이제 relative delta가 들어감

    for i in range(len(images) - 1):
        anchor_idx = i
        positive_idx = i + 1

        # ====================== 이미지 처리 (크롭 + resize) ======================
        apply_crop = random.random() < crop_prob
        crop_box = None
        pair_imgs = []

        for idx in [anchor_idx, positive_idx]:
            with Image.open(images[idx]) as im:
                im = im.convert("L")
                orig_w, orig_h = im.size
                crop_w, crop_h = RESIZE_HW

                if apply_crop:
                    if crop_box is None:
                        left = (orig_w - crop_w) / 2
                        top = (orig_h - crop_h) / 2          # ← 여기 orig_h로 수정 (버그 수정)
                        crop_box = (left, top, left + crop_w, top + crop_h)

                    cropped = im.crop(crop_box)
                    resized = cropped.resize(RESIZE_HW, Image.BILINEAR)
                else:
                    resized = im.resize(RESIZE_HW, Image.BILINEAR)

                arr = np.array(resized, dtype=np.uint8)
                pair_imgs.append(arr)

        # ====================== Relative Label 생성 (가장 중요) ======================
        anchor_label = labels[anchor_idx]      # [num_labels]
        positive_label = labels[positive_idx]

        # delta = positive - anchor
        delta_label = positive_label - anchor_label   # shape 동일하게 유지

        # pair 생성
        pair = np.stack(pair_imgs)                    # [2, H, W]

        pairs.append(pair)
        pair_labels.append(delta_label)

    # 최종 배열로 변환
    Xs = np.stack(pairs)           # [num_pairs, 2, H, W]   또는 [num_pairs, 2, H, W, 1] 필요시
    Ys = np.stack(pair_labels)     # [num_pairs, num_labels]  ← 이제 relative delta

    print(f"[make_pairs_trainval] {d} → {len(pairs)} pairs generated | "
          f"crop_prob={crop_prob*100:.0f}% | Relative Delta Label 사용")

    return Xs, Ys


def make_pairs_from_dir_test(d, label_names, num_pairs_per_dir=None):
    """
    Test/Validation용 pair 생성 함수
    - non-overlapping pair (0-1, 2-3, 4-5, ...)
    - 레이블을 positive - anchor 의 relative delta로 변경
    """
    # 이미지 목록 로드
    images = list_sorted_images(d)
    if len(images) < 2:
        return None, None

    # 라벨 로드
    labels = load_label_excel(d, EXCEL_NAME, label_names)  # [num_frames, num_labels]

    pairs = []
    pair_labels = []   # ← 이제 relative delta 저장

    # ==================== overlapping pair 생성 ====================
    for i in range(0, len(images) - 1):   # step=1
        anchor_idx = i
        positive_idx = i + 1

        # 이미지 로드 및 resize
        anchor_img = np.array(Image.open(images[anchor_idx]).resize(RESIZE_HW))
        positive_img = np.array(Image.open(images[positive_idx]).resize(RESIZE_HW))

        # ====================== Relative Delta Label 생성 ======================
        anchor_label = labels[anchor_idx]
        positive_label = labels[positive_idx]

        delta_label = positive_label - anchor_label        # ← 핵심 변경

        # pair 생성 [2, H, W] 또는 [2, H, W, C]
        pair = np.stack([anchor_img, positive_img])

        pairs.append(pair)
        pair_labels.append(delta_label)

    # 최종 배열로 변환
    Xs = np.stack(pairs)           # [num_pairs, 2, H, W, C] 또는 [num_pairs, 2, H, W]
    Ys = np.stack(pair_labels)     # [num_pairs, num_labels]  ← relative delta

    print(f"[make_pairs_test] {d} → {len(pairs)} pairs generated | "
          f"non-overlapping (0-1, 2-3, ...) | Relative Delta Label 사용")

    return Xs, Ys




