import torch

class Config:
    TOTAL_MAIN_NUMS = 35       # 01 -> 35
    TOTAL_SPECIAL_NUMS = 12    # 01 -> 12
    NUM_MAIN_PICKS = 5

    # 47 (35 main + 12 special) + 1 (is_split flag) = 48
    INPUT_DIM = 48

    WINDOW_SIZE = 30
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30
    VAL_RATIO = 0.15

    # Cấu hình chiến lược chia hũ
    SPLIT_TOP_MAIN_POOL = 10   # Chọn Top 10 số chính tiềm năng nhất để tạo dàn
    SPLIT_TOP_SPECIAL_POOL = 3 # Chọn Top 3 số đặc biệt

    DATA_PATH = "data/raw/lotto_535.csv"
    MODEL_SAVE_PATH = "models/checkpoint.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"