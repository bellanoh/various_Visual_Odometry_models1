
# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from dataset import PressSequenceDataset
from utils_siam2 import SeqTransform, save_checkpoint_full, denormalize_labels
from torch.nn.utils.rnn import pad_sequence
import torch
import time



# ====================== 설정 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_WARMUP = 20
NUM_ITER = 200
MAX_KEYPOINTS = 512

print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Keypoints: {MAX_KEYPOINTS}\n")

# ====================== 모델 로드 ======================
model = siamVONet().to(DEVICE)

# ====================== FPS 측정 준비 ======================
model.eval()

dummy_anchor   = torch.rand(BATCH_SIZE, 1, 32, 32, device=DEVICE, dtype=torch.float32)   # Grayscale
dummy_positive = torch.rand(BATCH_SIZE, 1, 32, 32, device=DEVICE, dtype=torch.float32)
dummy_altitude = torch.rand(BATCH_SIZE, 1, 32, 32, device=DEVICE, dtype=torch.float32)
dummy_yaw      = torch.rand(BATCH_SIZE, 1, 32, 32, device=DEVICE, dtype=torch.float32)

NUM_WARMUP = 20
NUM_ITER   = 100      # FPS 측정 반복 횟수 (필요시 200~300까지 올려도 됨)

# ====================== Warm-up ======================
print(f"Starting Warm-up ({NUM_WARMUP} iterations)...")
with torch.no_grad():
    for _ in range(NUM_WARMUP):
        _ = model(dummy_anchor, dummy_positive, dummy_altitude, dummy_yaw)
        torch.cuda.synchronize()   # GPU 동기화 (정확한 측정을 위해)

torch.cuda.synchronize()
print("Warm-up completed.")

# ====================== FPS 측정 ======================
times = []
peak_memories = []

with torch.no_grad():
    for i in range(NUM_ITER):
        start = time.perf_counter()

        _ = model(dummy_anchor, dummy_positive, dummy_altitude, dummy_yaw)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)

        # 메모리 측정
        if DEVICE.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)  # MB
            peak_memories.append(peak_mem)

# ====================== 결과 ======================
avg_time = np.mean(times)
fps_batch = 1.0 / avg_time
fps_images = fps_batch * BATCH_SIZE * 2   # anchor + positive = 2장

print("=" * 80)
print("DinoVOModel FPS Measurement Result")
print("=" * 80)
print(f"Average time per batch     : {avg_time * 1000:.2f} ms")
print(f"FPS (batches per second)   : {fps_batch:.2f}")
print(f"FPS (images per second)    : {fps_images:.2f}   ← anchor + positive")
print(f"Approx. sequences/sec      : {fps_batch:.2f}")
print("-" * 80)

if DEVICE.type == "cuda":
    avg_peak_mem = np.mean(peak_memories)
    max_peak_mem = max(peak_memories)
    print(f"Average Peak Memory        : {avg_peak_mem:.1f} MB")
    print(f"Max Peak Memory            : {max_peak_mem:.1f} MB")

print("=" * 80)

import torch
state_dict = torch.load("results/best_model.pth", map_location="cpu", weights_only=True)

total_params = sum(p.numel() for p in state_dict.values())
print(f"Total parameters in .pth file: {total_params:,}")
print(f"Model size ≈ {total_params / 1_000_000:.2f} M parameters")
print(f"Approx. file size (float32): {total_params * 4 / (1024**2):.2f} MB")

