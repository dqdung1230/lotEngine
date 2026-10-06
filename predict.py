import argparse
import torch
import numpy as np
import pandas as pd
from src.config import Config
from src.dataset import LottoDataset
from models.lotto_lstm import ConditionalLottoLSTM
from src.split_strategy import generate_split_draw_tickets

def predict(is_split_draw=False, num_tickets=60):
    # --- ĐỌC THÔNG TIN KỲ QUAY ---
    df = pd.read_csv(Config.DATA_PATH)
    start_draw = df.iloc[0]['draw_id']
    end_draw = df.iloc[-1]['draw_id']
    next_draw = end_draw + 1

    print("=" * 60)
    print(f" THÔNG TIN DỮ LIỆU:")
    print(f" - Dữ liệu huấn luyện: Từ kỳ {start_draw} đến kỳ {end_draw} (Tổng: {len(df)} kỳ)")
    print(f" - Đang dự đoán cho  : KỲ {next_draw}")
    print("=" * 60)
    # ----------------------------------------------

    dataset = LottoDataset(Config.DATA_PATH)
    latest_window = dataset.X[-1].unsqueeze(0).to(Config.DEVICE)

    split_tensor = torch.tensor([[1.0 if is_split_draw else 0.0]], dtype=torch.float32).to(Config.DEVICE)

    model = ConditionalLottoLSTM().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
    model.eval()

    with torch.no_grad():
        logits_m, logits_s = model(latest_window, split_tensor)
        prob_m = torch.sigmoid(logits_m).squeeze().cpu().numpy()
        prob_s = torch.softmax(logits_s, dim=-1).squeeze().cpu().numpy()

    print("=" * 60)
    if not is_split_draw:
        # --- ĐÃ SỬA: ÁP DỤNG LẤY MẪU NGẪU NHIÊN THEO TRỌNG SỐ ---
        # 1. Chuẩn hóa phân phối xác suất
        prob_m_norm = prob_m / np.sum(prob_m)
        prob_s_norm = prob_s / np.sum(prob_s)

        # 2. Bốc 5 số chính dựa trên tỷ lệ AI (không trùng lặp)
        sampled_main = np.random.choice(
            np.arange(1, Config.TOTAL_MAIN_NUMS + 1),
            size=Config.NUM_MAIN_PICKS,
            replace=False,
            p=prob_m_norm
        )
        top5_main = sorted(sampled_main)

        # 3. Bốc 1 số đặc biệt dựa trên tỷ lệ AI
        best_special = int(np.random.choice(np.arange(1, Config.TOTAL_SPECIAL_NUMS + 1), p=prob_s_norm))

        print(" CHẾ ĐỘ: KỲ QUAY TIÊU CHUẨN (LẤY MẪU THEO TRỌNG SỐ AI)")
        print(f" 5 số chính dự đoán   : {[f'{x:02d}' for x in top5_main]}")
        print(f" 1 số đặc biệt dự đoán : {best_special:02d}")

    else:
        # --- KỲ CHIA HŨ (GIỮ NGUYÊN) ---
        print(f" CHẾ ĐỘ: KỲ CHIA HŨ (TẠO DÀN {num_tickets} VÉ TỐI ƯU)")
        csv_file = "danh_sach_ve_chia_hu.csv"
        pool, df_tickets = generate_split_draw_tickets(
            prob_m, prob_s,
            pool_size=16,
            num_tickets=num_tickets,
            output_csv=csv_file
        )

        print(f"\n Đã tạo thành công {len(df_tickets)} vé và xuất ra file: {csv_file}")
        print("\n--- XEM TRƯỚC 10 VÉ ĐẦU TIÊN TRONG DÀN ---")
        for _, row in df_tickets.head(10).iterrows():
            print(f" Vé #{row['STT']:02d}: [{row['So_1']}, {row['So_2']}, {row['So_3']}, {row['So_4']}, {row['So_5']}] | ĐB: {row['Dac_Biet']}")
        print(f" ... và {len(df_tickets) - 10} vé còn lại trong file CSV.")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Bật cờ dự đoán cho kỳ chia hũ")
    parser.add_argument("--tickets", type=int, default=60, help="Số lượng vé muốn mua (50 - 80)")
    args = parser.parse_args()

    predict(is_split_draw=args.split, num_tickets=args.tickets)