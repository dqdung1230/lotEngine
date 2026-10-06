lotto_engine/
├── data/
│   └── raw/
│       └── lotto_535.csv          # Dữ liệu 500 kỳ quay mẫu sẵn sàng chạy thử
├── models/
│   ├── __init__.py
│   └── lotto_lstm.py              # Mô hình LSTM Multi-task (5 số chính + 1 số đặc biệt)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Siêu tham số & cấu hình hệ thống
│   ├── dataset.py                 # Sliding window dataset & vector encoding
│   ├── feature_engineering.py     # Thống kê tần suất & chu kỳ trễ (gap)
│   └── metrics.py                 # Hàm tính Hit-rate & đánh giá
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── train.py                       # Script huấn luyện và lưu checkpoint
└── predict.py                     # Script suy luận dự đoán kỳ tiếp theo