import torch
import torch.nn as nn
from src.config import Config

class ConditionalLottoLSTM(nn.Module):
    def __init__(self):
        super(ConditionalLottoLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            batch_first=True,
            dropout=Config.DROPOUT if Config.NUM_LAYERS > 1 else 0.0
        )

        # Nhận feature từ LSTM (128) + 1 chiều cờ is_split của kỳ mục tiêu = 129
        self.shared_fc = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM + 1, 64),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )

        self.main_head = nn.Linear(64, Config.TOTAL_MAIN_NUMS)
        self.special_head = nn.Linear(64, Config.TOTAL_SPECIAL_NUMS)

    def forward(self, x, is_split_target):
        # x: [batch, window_size, 48]
        # is_split_target: [batch, 1]
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :] # [batch, hidden_dim]

        # Ghép điều kiện chia hũ của kỳ cần dự đoán
        conditioned_features = torch.cat([last_step, is_split_target], dim=1)
        features = self.shared_fc(conditioned_features)

        logits_main = self.main_head(features)
        logits_special = self.special_head(features)

        return logits_main, logits_special