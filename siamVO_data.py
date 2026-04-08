# -*- coding: utf-8 -*-


import os
import sys
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm


# =========================================
# Siamese Pair Generation
# =========================================
def make_pairs_from_dir_trainval(d, label_names, add_NAMES):
    """
    연속된 프레임으로 overlapping pair 생성 (한 바퀴만 돌림)
    - pose label은 relative delta
    - altitude, yaw는 positive 값 그대로 사용
    """
    images = list_sorted_images(d)
    if len(images) < 2:
        print(f"Warning: Not enough images in directory '{d}'")
        return None, None

    # 라벨 로드
    labels = load_label_excel(d, EXCEL_NAME, label_names)  # [num_frames, num_labels]
    aly = load_aly_excel(d, EXCEL_NAME, add_NAMES)  # [num_frames, 2]

    pairs = []
    pair_labels = []

    # ==================== overlapping pair 한 바퀴 ====================
    for i in range(len(images) - 1):  # 0 ~ len-2
        anchor_idx = i
        positive_idx = i + 1

        # 이미지 처리
        anchor_img = np.array(Image.open(images[anchor_idx]).convert('RGB').resize(RESIZE_HW))
        positive_img = np.array(Image.open(images[positive_idx]).convert('RGB').resize(RESIZE_HW))

        # Grayscale 변환
        anchor_img = np.mean(anchor_img, axis=-1, keepdims=True).astype(np.float32)
        positive_img = np.mean(positive_img, axis=-1, keepdims=True).astype(np.float32)

        # altitude & yaw (positive 값 그대로)
        altitude = np.full((32, 32, 1), aly[positive_idx, 0], dtype=np.float32)
        yaw = np.full((32, 32, 1), aly[positive_idx, 1], dtype=np.float32)

        # delta pose (relative difference)
        delta_pose = labels[positive_idx] - labels[anchor_idx]

        # pair 저장
        pair_labels.append(delta_pose)

        # Stack: [anchor_gray, positive_gray, altitude_map, yaw_map]
        single_pair = np.stack([anchor_img, positive_img, altitude, yaw], axis=0)  # (4, 32, 32, 1)
        pairs.append(single_pair)

    # 최종 반환
    Xs = np.stack(pairs)  # [num_pairs, 4, 32, 32, 1]
    Ys = np.stack(pair_labels)  # [num_pairs, num_labels]

    print(f"[make_pairs_trainval] {d} → {len(pairs)} overlapping pairs generated | "
          f"alt/yaw = positive | pose = relative delta")

    return Xs, Ys


def make_pairs_from_dir_test(d, label_names, add_NAMES):
    """
    Test용 overlapping pair 생성 (trainval과 동일한 방식)
    - pose label은 relative delta
    - altitude, yaw는 positive 값 그대로 사용
    """
    images = list_sorted_images(d)
    if len(images) < 2:
        print(f"Warning: Not enough images in directory '{d}'")
        return None, None

    # 라벨 로드
    labels = load_label_excel(d, EXCEL_NAME, label_names)
    aly = load_aly_excel(d, EXCEL_NAME, add_NAMES)

    pairs = []
    pair_labels = []

    # ==================== overlapping pair (test도 동일하게) ====================
    for i in range(len(images) - 1):   # 0 ~ len-2
        anchor_idx = i
        positive_idx = i + 1

        # 이미지 처리
        anchor_img = np.array(Image.open(images[anchor_idx]).convert('RGB').resize(RESIZE_HW))
        positive_img = np.array(Image.open(images[positive_idx]).convert('RGB').resize(RESIZE_HW))

        # Grayscale 변환
        anchor_img = np.mean(anchor_img, axis=-1, keepdims=True).astype(np.float32)
        positive_img = np.mean(positive_img, axis=-1, keepdims=True).astype(np.float32)

        # altitude & yaw (positive 값 그대로)
        altitude = np.full((32, 32, 1), aly[positive_idx, 0], dtype=np.float32)
        yaw = np.full((32, 32, 1), aly[positive_idx, 1], dtype=np.float32)

        # delta pose (relative difference)
        delta_pose = labels[positive_idx] - labels[anchor_idx]

        pair_labels.append(delta_pose)

        # Stack: [anchor_gray, positive_gray, altitude_map, yaw_map]
        single_pair = np.stack([anchor_img, positive_img, altitude, yaw], axis=0)  # (4, 32, 32, 1)
        pairs.append(single_pair)

    Xs = np.stack(pairs)      # [num_pairs, 4, 32, 32, 1]
    Ys = np.stack(pair_labels)

    print(f"[make_pairs_test] {d} → {len(pairs)} overlapping pairs generated | "
          f"alt/yaw = positive | pose = relative delta")

    return Xs, Ys


