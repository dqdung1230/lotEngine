import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from src.config import Config

class LottoDataset(Dataset):
    def __init__(self, csv_path, window_size=Config.WINDOW_SIZE):
        df = pd.read_csv(csv_path)

        # Nếu chưa có cột is_split, tự khởi tạo mặc định là 0
        if 'is_split' not in df.columns:
            df['is_split'] = 0

        main_cols = ['n1', 'n2', 'n3', 'n4', 'n5']
        self.main_data = df[main_cols].values
        self.special_data = df['special'].values
        self.split_data = df['is_split'].values

        self.window_size = window_size
        self.encoded_draws = self._encode_all()
        self.X, self.target_is_split, self.y_main, self.y_special = self._build_sliding_windows()

    def _encode_all(self):
        total = len(self.main_data)
        # Vector 48 chiều: [35 main, 12 special, 1 is_split]
        encoded = np.zeros((total, Config.INPUT_DIM), dtype=np.float32)
        for i in range(total):
            for n in self.main_data[i]:
                encoded[i, n - 1] = 1.0
            spec = self.special_data[i]
            encoded[i, Config.TOTAL_MAIN_NUMS + (spec - 1)] = 1.0
            encoded[i, -1] = float(self.split_data[i])
        return encoded

    def _build_sliding_windows(self):
        X, target_split, y_main, y_special = [], [], [], []
        total_draws = len(self.encoded_draws)
        for i in range(total_draws - self.window_size):
            X.append(self.encoded_draws[i : i + self.window_size])
            target_split.append([float(self.split_data[i + self.window_size])])
            y_main.append(self.encoded_draws[i + self.window_size, :Config.TOTAL_MAIN_NUMS])
            y_special.append(self.special_data[i + self.window_size] - 1)

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(target_split), dtype=torch.float32),
            torch.tensor(np.array(y_main), dtype=torch.float32),
            torch.tensor(np.array(y_special), dtype=torch.long)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.target_is_split[idx], self.y_main[idx], self.y_special[idx]