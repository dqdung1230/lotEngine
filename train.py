import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.config import Config
from src.dataset import LottoDataset
from models.lotto_lstm import ConditionalLottoLSTM

def run_training():
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    dataset = LottoDataset(Config.DATA_PATH)

    # Chia tập theo trình tự thời gian
    split_idx = int(len(dataset) * (1 - Config.VAL_RATIO))
    train_loader = DataLoader(Subset(dataset, range(split_idx)), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(split_idx, len(dataset))), batch_size=Config.BATCH_SIZE, shuffle=False)

    # Khởi tạo mô hình Conditional
    model = ConditionalLottoLSTM().to(Config.DEVICE)
    use_split_flag = True
    # --- ĐOẠN CODE MỚI: KIỂM TRA VÀ HỌC TIẾP (FINE-TUNING) ---
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"[*] Đã tìm thấy trí nhớ cũ tại {Config.MODEL_SAVE_PATH}.")
        print("[*] Tiến hành nạp kiến thức để HỌC TIẾP (Fine-tuning)...")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))

        # Giảm tốc độ học (Learning Rate) xuống 10 lần để không làm hỏng kiến thức cũ
        current_lr = Config.LEARNING_RATE / 10
        current_epochs = 10 # Chỉ cần học thêm 10 vòng là đủ ngấm dữ liệu mới
    else:
        print("[*] Chưa có trí nhớ cũ. Sẽ học mới hoàn toàn từ đầu...")
        current_lr = Config.LEARNING_RATE
        current_epochs = Config.EPOCHS

    crit_main = nn.BCEWithLogitsLoss()
    crit_special = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=Config.WEIGHT_DECAY)

    best_val_loss = float('inf')

    print(f"Bắt đầu huấn luyện trên thiết bị: {Config.DEVICE}")
    for epoch in range(1, current_epochs + 1):
        model.train()
        train_loss = 0.0

        # Nhận đủ 4 biến: X, split_flag, ym, ys
        for batch in train_loader:
            if len(batch) == 4:
                X, split_flag, ym, ys = batch
                split_flag = split_flag.to(Config.DEVICE)
            else:
                X, ym, ys = batch
                split_flag = None

            X, ym, ys = X.to(Config.DEVICE), ym.to(Config.DEVICE), ys.to(Config.DEVICE)
            optimizer.zero_grad()

            if use_split_flag and split_flag is not None:
                out_m, out_s = model(X, split_flag)
            else:
                out_m, out_s = model(X)

            loss = crit_main(out_m, ym) + crit_special(out_s, ys)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Đánh giá trên tập Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    X, split_flag, ym, ys = batch
                    split_flag = split_flag.to(Config.DEVICE)
                else:
                    X, ym, ys = batch
                    split_flag = None

                X, ym, ys = X.to(Config.DEVICE), ym.to(Config.DEVICE), ys.to(Config.DEVICE)

                if use_split_flag and split_flag is not None:
                    out_m, out_s = model(X, split_flag)
                else:
                    out_m, out_s = model(X)

                val_loss += (crit_main(out_m, ym) + crit_special(out_s, ys)).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

        print(f"Epoch [{epoch:02d}/{Config.EPOCHS}] - Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    print(f"\nĐã lưu checkpoint tốt nhất vào: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    run_training()