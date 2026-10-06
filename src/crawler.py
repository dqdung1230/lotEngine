import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from src.config import Config

def fetch_latest_draw_from_web():
    """
    Hàm này thực hiện kết nối đến trang web xổ số và bóc tách dữ liệu.
    LƯU Ý: URL và các thẻ HTML (class/id) dưới đây là ví dụ cấu trúc chung.
    Bạn cần F12 (Inspect) trang web thực tế của bạn để thay đổi tên class cho đúng.
    """
    # Thay bằng URL trang web bạn muốn lấy kết quả
    url = "https://minhngoc.net.vn/ket-qua-xo-so/demo-5-35.html"

    try:
        # Gửi request đóng giả làm trình duyệt web
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # -- ĐOẠN NÀY PHỤ THUỘC VÀO CẤU TRÚC HTML CỦA WEB --
        # Ví dụ web có 1 block tên là: <div class="draw-result" data-id="501" data-date="2026-09-08">
        # Và các quả bóng: <span class="ball">15</span>

        # Giả lập dữ liệu trả về sau khi parse HTML thành công:
        latest_data = {
            "draw_id": 501,
            "date": "2026-09-08",
            "numbers": [12, 18, 22, 29, 31], # 5 số chính đã sắp xếp
            "special": 7,                    # 1 số đặc biệt
            "is_split": 0                    # 1 nếu phát hiện text "Kỳ chia hũ" trên web, ngược lại 0
        }

        return latest_data

    except Exception as e:
        print(f"[!] Lỗi khi cào dữ liệu từ web: {e}")
        return None

def update_dataset():
    csv_path = Config.DATA_PATH

    # 1. Đọc file dữ liệu hiện tại để tìm kỳ cuối cùng
    if not os.path.exists(csv_path):
        print("[!] Không tìm thấy file dữ liệu gốc!")
        return

    df = pd.read_csv(csv_path)
    last_draw_id = int(df.iloc[-1]['draw_id'])

    print(f"[*] Kỳ quay gần nhất trong máy: {last_draw_id}")
    print("[*] Đang kết nối web để lấy kết quả mới...")

    # 2. Cào dữ liệu mới
    new_data = fetch_latest_draw_from_web()
    if not new_data:
        return

    # 3. Kiểm tra xem kết quả này đã có trong máy chưa
    if new_data['draw_id'] <= last_draw_id:
        print(f"[*] Web chưa cập nhật kỳ mới (Kỳ trên web hiện tại: {new_data['draw_id']}).")
        return

    if new_data['draw_id'] > last_draw_id + 1:
        print(f"[!] Cảnh báo: Bạn đang bị hụt mất dữ liệu từ kỳ {last_draw_id + 1} đến {new_data['draw_id'] - 1}!")
        # Bạn có thể viết thêm vòng lặp ở đây để cào bù các kỳ bị thiếu

    # 4. Nối dữ liệu mới vào file CSV
    new_row = {
        'draw_id': new_data['draw_id'],
        'date': new_data['date'],
        'n1': new_data['numbers'][0],
        'n2': new_data['numbers'][1],
        'n3': new_data['numbers'][2],
        'n4': new_data['numbers'][3],
        'n5': new_data['numbers'][4],
        'special': new_data['special'],
        'is_split': new_data['is_split']
    }

    df_new = pd.DataFrame([new_row])
    # mode='a' (append) giúp ghi thêm vào cuối file mà không xóa dữ liệu cũ
    df_new.to_csv(csv_path, mode='a', header=False, index=False)

    print(f"[+] Thành công! Đã thêm kỳ {new_data['draw_id']} vào tập dữ liệu học.")

if __name__ == "__main__":
    update_dataset()