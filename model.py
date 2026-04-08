# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import ResNet18_Weights, ResNet101_Weights, ResNet152_Weights


class CNNBiLSTM(nn.Module):

    def __init__(self, hidden_dim=128, num_layers=1, out_dim=3):
        super().__init__()
       #----- [1] CNN Backbone -----#
        backbone = models.resnet50(
            weights=ResNet50_Weights.DEFAULT)  # ResNet50 추천 – 더 깊은 ResNet101/152도 가능 but 메모리 많이 먹음
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1],  # (B, 2048, 1, 1) – ResNet50 출력 채널 2048
            nn.Flatten(start_dim=1)  # (B, 2048)
        )
        self.feature_dim = 2048  # ResNet50에 맞춰 업데이트 (ResNet18은 512)

        # backbone = models.resnet18(
        #     weights=ResNet18_Weights.DEFAULT)  # Lightweight, good for smaller datasets or less compute
        # self.feature_extractor = nn.Sequential(
        #     *list(backbone.children())[:-1],  # (B, 512, 1, 1) – ResNet18 출력 채널 512
        #     nn.Flatten(start_dim=1)  # (B, 512)
        # )
        # self.feature_dim = 512  # ResNet18에 맞춰 업데이트

        # # # ----- [1] CNN Backbone -----
        # backbone = models.resnet101(
        #     weights=ResNet101_Weights.DEFAULT)  # Deeper than 50, better for complex tasks but more memory-intensive
        # self.feature_extractor = nn.Sequential(
        #     *list(backbone.children())[:-1],  # (B, 2048, 1, 1) – ResNet101 출력 채널 2048
        #     nn.Flatten(start_dim=1)  # (B, 2048)
        # )
        # self.feature_dim = 2048  # ResNet101에 맞춰 업데이트



        # # ----- [1] CNN Backbone -----
        # backbone = models.resnet152(
        #     weights=ResNet152_Weights.DEFAULT)  # Deepest standard ResNet, highest accuracy potential but heavy on resources
        # self.feature_extractor = nn.Sequential(
        #     *list(backbone.children())[:-1],  # (B, 2048, 1, 1) – ResNet152 출력 채널 2048
        #     nn.Flatten(start_dim=1)  # (B, 2048)
        # )
        # self.feature_dim = 2048  # ResNet152에 맞춰 업데이트

        # ----- [2] BiLSTM -----
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # ----- [3] Linear Head -----
        self.fc = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        if C == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale → RGB

        feats = self.feature_extractor(x)   # (B*T, 512)
        feats = feats.view(B, T, -1)        # (B, T, 512)

        lstm_out, _ = self.lstm(feats)      # (B, T, 2H)
        # out = self.fc(lstm_out)               # (B, out_dim)
        pooled = torch.mean(lstm_out, dim=1)  # temporal mean pooling
        out = self.fc(pooled)  # (B, out_dim)
        return out
