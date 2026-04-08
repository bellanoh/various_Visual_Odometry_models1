# -*- coding: utf-8 -*-

# =========================================
# Siamese Pair Generation
# =========================================
import numpy as np
from PIL import Image
import random

def make_pairs_from_dir_trainval(d, label_names, num_keypoints=512, crop_prob=0.0):
    """
    연속된 프레임으로 overlapping pair 생성 + Relative Delta Label
    """
    images = list_sorted_images(d)
    if len(images) < 2:
        print(f"Warning: Not enough images in directory '{d}'")
        return np.empty((0, 2, *RESIZE_HW, 1), dtype=np.uint8), \
               np.empty((0, len(label_names)), dtype=np.float32), \
               []

    labels = load_label_excel(d, EXCEL_NAME, label_names)

    pairs = []
    pair_labels = []
    keypoints_list = []

    for i in range(len(images) - 1):
        anchor_idx = i
        positive_idx = i + 1

        apply_crop = random.random() < crop_prob
        crop_box = None
        pair_imgs = []

        for idx in [anchor_idx, positive_idx]:
            with Image.open(images[idx]) as im:
                orig_w, orig_h = im.size
                crop_w, crop_h = RESIZE_HW

                if apply_crop:
                    if crop_box is None:
                        left = (orig_w - crop_w) / 2
                        top = (orig_h - crop_h) / 2          # height 기준 수정
                        crop_box = (left, top, left + crop_w, top + crop_h)

                    cropped = im.crop(crop_box)
                    resized = cropped.resize(RESIZE_HW, Image.BILINEAR)
                else:
                    resized = im.resize(RESIZE_HW, Image.BILINEAR)

                # Grayscale 변환
                arr = np.array(resized)
                if arr.ndim == 3:
                    gray = np.mean(arr, axis=-1, keepdims=True).astype(np.uint8)
                else:
                    gray = arr[..., np.newaxis].astype(np.uint8)

                pair_imgs.append(gray)

        frames = np.stack(pair_imgs)                    # [2, H, W, 1]

        # ====================== Salient Keypoint Detection ======================
        kp_anchor, score_anchor = detect_salient_keypoints(
            frames[0], num_keypoints=num_keypoints, grad_threshold=0.01, rP=14, rNMS=8
        )
        kp_positive, score_positive = detect_salient_keypoints(
            frames[1], num_keypoints=num_keypoints, grad_threshold=0.01, rP=14, rNMS=8
        )

        # ====================== Relative Delta Label ======================
        delta_label = labels[positive_idx] - labels[anchor_idx]

        pairs.append(frames)
        pair_labels.append(delta_label)
        keypoints_list.append({
            'anchor': kp_anchor,
            'positive': kp_positive,
            'anchor_score': score_anchor,
            'positive_score': score_positive
        })

    Xs = np.stack(pairs)                    # [num_pairs, 2, H, W, 1]
    Ys = np.stack(pair_labels)

    print(f"[make_pairs_trainval] {d} → {len(pairs)} consecutive overlapping pairs | "
          f"crop_prob={crop_prob*100:.0f}% | relative delta label | num_keypoints={num_keypoints}")

    return Xs, Ys, keypoints_list


# ====================== make_pairs_from_dir_test ======================
def make_pairs_from_dir_test(d, label_names, num_keypoints=512, crop_prob=0.0):
    """
    Test용 overlapping pair 생성 + Relative Delta Label
    """
    images = list_sorted_images(d)
    if len(images) < 2:
        print(f"Warning: Not enough images in directory '{d}'")
        return np.empty((0, 2, *RESIZE_HW, 1), dtype=np.uint8), \
            np.empty((0, len(label_names)), dtype=np.float32), \
            []

    labels = load_label_excel(d, EXCEL_NAME, label_names)

    pairs = []
    pair_labels = []
    keypoints_list = []

    for i in range(len(images) - 1):
        anchor_idx = i
        positive_idx = i + 1
        if positive_idx >= len(images):
            break

        apply_crop = random.random() < crop_prob
        crop_box = None
        pair_imgs = []

        for idx in [anchor_idx, positive_idx]:
            with Image.open(images[idx]) as im:
                orig_w, orig_h = im.size
                crop_w, crop_h = RESIZE_HW

                if apply_crop:
                    if crop_box is None:
                        left = (orig_w - crop_w) / 2
                        top = (orig_h - crop_h) / 2
                        crop_box = (left, top, left + crop_w, top + crop_h)

                    cropped = im.crop(crop_box)
                    resized = cropped.resize(RESIZE_HW, Image.BILINEAR)
                else:
                    resized = im.resize(RESIZE_HW, Image.BILINEAR)

                arr = np.array(resized)
                if arr.ndim == 3:
                    gray = np.mean(arr, axis=-1, keepdims=True).astype(np.uint8)
                else:
                    gray = arr[..., np.newaxis].astype(np.uint8)

                pair_imgs.append(gray)

        frames = np.stack(pair_imgs)  # [2, H, W, 1]

        # Keypoint Detection
        kp_anchor, score_anchor = detect_salient_keypoints(
            frames[0], num_keypoints=num_keypoints, grad_threshold=0.01, rP=14, rNMS=8
        )
        kp_positive, score_positive = detect_salient_keypoints(
            frames[1], num_keypoints=num_keypoints, grad_threshold=0.01, rP=14, rNMS=8
        )

        # ====================== Relative Delta Label ======================
        delta_label = labels[positive_idx] - labels[anchor_idx]

        pairs.append(frames)
        pair_labels.append(delta_label)
        keypoints_list.append({
            'anchor': kp_anchor,
            'positive': kp_positive,
            'anchor_score': score_anchor,
            'positive_score': score_positive
        })

    if len(pairs) == 0:
        print(f"[make_pairs_test] {d} → No pairs generated")
        return np.empty((0, 2, *RESIZE_HW, 1), dtype=np.uint8), \
            np.empty((0, len(label_names)), dtype=np.float32), \
            []

    Xs = np.stack(pairs)
    Ys = np.stack(pair_labels)

    print(f"[make_pairs_test] {d} → {len(pairs)} non-overlapping pairs | "
          f"crop_prob={crop_prob * 100:.0f}% | relative delta label")

    return Xs, Ys, keypoints_list
