import itertools
import numpy as np
import pandas as pd
from collections import defaultdict

def generate_split_draw_tickets(prob_main, prob_special, pool_size=16, num_tickets=60, output_csv="danh_sach_ve_chia_hu.csv"):
    """
    Sinh dàn N vé đơn (50-80 vé) tối ưu độ phủ cho kỳ chia hũ:
    - pool_size: Số lượng bóng chính trong pool rút gọn (mặc định 16 số).
    - num_tickets: Số lượng vé muốn mua (ví dụ: 50, 60, 80 vé).
    """
    # 1. Trích xuất Pool Top M số chính và Top 4 số đặc biệt
    top_main_indices = np.argsort(prob_main)[-pool_size:][::-1]
    pool_main = sorted([idx + 1 for idx in top_main_indices])

    top_special_indices = np.argsort(prob_special)[-4:][::-1]
    top_specials = [idx + 1 for idx in top_special_indices]
    special_weights = prob_special[top_special_indices]
    special_probs_norm = special_weights / np.sum(special_weights)

    print(f"[*] Pool {pool_size} số chính được AI chọn: {pool_main}")
    print(f"[*] Top số đặc biệt ưu tiên: {top_specials} (Tỷ lệ: {np.round(special_probs_norm, 2)})")

    # 2. Sinh tất cả tổ hợp chập 5 từ Pool
    all_combos = list(itertools.combinations(pool_main, 5))

    # 3. Chấm điểm AI cơ bản cho từng tổ hợp
    combo_scores = {}
    for combo in all_combos:
        score = sum(prob_main[num - 1] for num in combo)
        combo_scores[combo] = score

    # 4. Thuật toán Tham lam (Greedy) tối ưu độ phủ các cặp số (2-match & 3-match)
    selected_combos = []
    pair_counts = defaultdict(int)

    # Lấy vé đầu tiên là vé có điểm AI cao nhất
    best_first_combo = max(combo_scores.keys(), key=lambda c: combo_scores[c])
    selected_combos.append(best_first_combo)
    for pair in itertools.combinations(best_first_combo, 2):
        pair_counts[pair] += 1

    remaining_combos = set(all_combos) - {best_first_combo}

    # Chọn tiếp các vé còn lại cho đến khi đủ số lượng vé yêu cầu
    while len(selected_combos) < num_tickets and remaining_combos:
        best_candidate = None
        best_candidate_score = -1e9

        for candidate in remaining_combos:
            ai_score = combo_scores[candidate]
            # Đếm số cặp số mới chưa xuất hiện nhiều trong dàn vé đã chọn
            overlap_penalty = sum(pair_counts[pair] for pair in itertools.combinations(candidate, 2))

            # Hàm tối ưu: Điểm AI cao + Phạt trùng lặp cặp số
            final_score = ai_score * 3.0 - (overlap_penalty * 0.5)

            if final_score > best_candidate_score:
                best_candidate_score = final_score
                best_candidate = candidate

        selected_combos.append(best_candidate)
        for pair in itertools.combinations(best_candidate, 2):
            pair_counts[pair] += 1
        remaining_combos.remove(best_candidate)

    # 5. Phân bổ số đặc biệt vào các vé
    tickets = []
    for i, main_nums in enumerate(selected_combos):
        # Gán số đặc biệt theo phân phối xác suất
        spec = np.random.choice(top_specials, p=special_probs_norm)
        tickets.append({
            "STT": i + 1,
            "So_1": f"{main_nums[0]:02d}",
            "So_2": f"{main_nums[1]:02d}",
            "So_3": f"{main_nums[2]:02d}",
            "So_4": f"{main_nums[3]:02d}",
            "So_5": f"{main_nums[4]:02d}",
            "Dac_Biet": f"{spec:02d}"
        })

    # 6. Xuất danh sách ra DataFrame và lưu file CSV
    df_tickets = pd.DataFrame(tickets)
    df_tickets.to_csv(output_csv, index=False, encoding='utf-8-sig')

    return pool_main, df_tickets