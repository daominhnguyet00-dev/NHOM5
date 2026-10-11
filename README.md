# 📈 Hệ Thống Kiểm Định Định Lượng: Chiến Lược Kết Hợp SMA & OBV

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Market: HOSE](https://img.shields.io/badge/Market-Vietnam%20HOSE-red.svg)](#)

Ứng dụng web tương tác được phát triển trên nền tảng **Streamlit**, nhằm kiểm định tính hiệu quả và độ bền vững của chiến lược giao dịch định lượng kết hợp giữa tín hiệu xu hướng **SMA (Simple Moving Average)** và dòng tiền khối lượng **OBV (On-Balance Volume)** trên thị trường chứng khoán Việt Nam (Sàn HOSE giai đoạn 2020 - 2023).

---

## 🌟 Tính Năng Nổi Bật

1. **Sàng Lọc & Xếp Hạng Kỹ Thuật (TA Scoring Framework)**:
   - Đánh giá toàn bộ 100 cổ phiếu trên sàn HOSE qua 4 nhóm nhân tố định lượng: **Trend** (SMA50 vs SMA200), **Momentum** (Return 6 tháng), **Volume** (Thanh khoản + OBV Confirmation), và **Risk** (Volatility + Max Drawdown).
   - Tùy biến trọng số theo các kịch bản: *Base (30/25/25/20)*, *Equal (25/25/25/25)*, *Risk-oriented (25/20/20/35)* để chọn ra Top cổ phiếu tiềm năng nhất (mặc định: `DIG`, `DPM`, `NLG`, `KBC`, `DCM`).

2. **Kiểm Định Chi Tiết Cổ Phiếu Đơn Lẻ (Single Stock Backtest)**:
   - Tùy chỉnh linh hoạt tham số `SMA Short`, `SMA Long`, và `OBV Window`.
   - Hỗ trợ chạy thuật toán tối ưu hóa tự động **Hyperopt TPE (Tree-structured Parzen Estimator)** để tìm bộ tham số tối đa hóa tỷ lệ Sharpe trên tập Train.
   - Hệ thống biểu đồ tương tác 4 tầng (Plotly): Đường giá & tín hiệu Mua/Bán, Chỉ báo OBV & MA, Trạng thái vị thế thị trường (Market Exposure), và Đường cong vốn (Equity Curve).

3. **Kiểm Định Danh Mục Đầu Tư Thực Chiến (Out-Of-Sample Test 2022)**:
   - Kiểm định nghiêm ngặt trên giai đoạn ngoài mẫu (**Test 2022** - năm thị trường sụt giảm mạnh/Bear Market) với cơ chế Warmup lịch sử đầy đủ.
   - Đối chiếu 3 phương pháp quản trị danh mục:
     1. **Equal Weight Buy & Hold** (Benchmark cơ sở).
     2. **SMA + OBV Equal Weight** (Phân bổ tỷ trọng đều).
     3. **SMA + OBV MPT Weight** (Tối ưu hóa tỷ trọng theo Modern Portfolio Theory).
   - Minh chứng khả năng phòng thủ vượt bậc: cắt giảm sụt giảm tài khoản (Drawdown) từ `-60.52%` xuống chỉ còn `-29.22%`.

4. **Phân Tích Độ Nhạy Chi Phí & Kiểm Tra Độ Bền (Robustness Checks)**:
   - Mô phỏng độ nhạy với các mức phí giao dịch thực tế: `0.0%`, `0.05%`, `0.10%`, và `0.20%`.
   - Tính toán chi phí quay vòng vị thế (Turnover cost) và quy tắc khớp lệnh trễ $T+1$ sát với thực tế thị trường Việt Nam.
   - Xuất báo cáo kết quả kiểm định ra file CSV.

---

## 📂 Cấu Trúc Thư Mục

```text
├── app.py                         # Mã nguồn chính ứng dụng web Streamlit
├── requirements.txt               # Danh sách thư viện phụ thuộc để deploy
├── HOSE_2020_2023_in.csv          # File dữ liệu giá lịch sử OHLCV các mã HOSE (2020 - 2023)
├── HOSE_STRATEGY_8_5_REVISED (1).ipynb # Notebook nghiên cứu gốc
└── README.md                      # Hướng dẫn chi tiết dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Trên Máy Cục Bộ (Local)

### 1. Yêu Cầu Hệ Thống
- Python version: `3.9`, `3.10`, hoặc `3.11`.
- Trình duyệt web hiện đại (Chrome, Edge, Firefox, Brave,...).

### 2. Các Bước Cài Đặt

1. **Clone repository về máy**:
   ```bash
   git clone https://github.com/<tai-khoan-cua-ban>/<ten-repo>.git
   cd <ten-repo>
   ```

2. **Tạo môi trường ảo (khuyến nghị)**:
   - Trên Windows:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Trên macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Cài đặt các gói phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Streamlit**:
   ```bash
   streamlit run app.py
   ```
   Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Deploy Lên Streamlit Community Cloud (Miễn Phí)

Để đưa ứng dụng lên mạng internet cho mọi người cùng sử dụng, bạn có thể deploy lên **Streamlit Community Cloud** chỉ với vài bước đơn giản:

### Bước 1: Đẩy Mã Nguồn Lên GitHub
1. Khởi tạo Git trong thư mục dự án (nếu chưa có):
   ```bash
   git init
   git add app.py requirements.txt README.md HOSE_2020_2023_in.csv
   git commit -m "feat: Khoi tao du an Streamlit Backtest SMA + OBV"
   ```
2. Tạo một repository mới trên GitHub (ví dụ: `hose-sma-obv-backtest`).
3. Đẩy code lên GitHub:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-cua-ban>/hose-sma-obv-backtest.git
   git push -u origin main
   ```

### Bước 2: Thiết Lập Trên Streamlit Cloud
1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc "Create app").
3. Điền các thông tin:
   - **Repository**: Chọn repo vừa tạo (ví dụ: `<tai-khoan-cua-ban>/hose-sma-obv-backtest`).
   - **Branch**: `main`.
   - **Main file path**: `app.py`.
4. Nhấn **"Deploy!"**.
5. Đợi hệ thống tự động build môi trường từ `requirements.txt` trong khoảng 1-2 phút. Sau khi hoàn tất, bạn sẽ nhận được đường link web app công khai để chia sẻ!

---

## 🔬 Phương Pháp Luận Định Lượng

### 1. Phân Tách Tập Dữ Liệu
- **In-Sample (Train)**: `2020-01-01` đến `2021-12-31` (Giai đoạn thị trường tăng trưởng mạnh).
- **Out-Of-Sample (Test)**: `2022-01-01` đến `2022-12-31` (Giai đoạn thị trường downtrend khốc liệt).
- **Nguyên tắc Warmup**: Giữ nguyên dữ liệu lịch sử trước năm 2022 khi tính các chỉ báo chu kỳ dài (SMA 200, SMA 350) để tránh hiện tượng mất tín hiệu đầu kỳ kiểm định.

### 2. Logic Tín Hiệu Kết Hợp (OR Logic)
- **Tín hiệu SMA**:
  $$\text{Mua khi: } SMA_{\text{short}} > SMA_{\text{long}} \quad \text{và} \quad SMA_{\text{short}, t-1} \le SMA_{\text{long}, t-1}$$
  $$\text{Bán khi: } SMA_{\text{short}} < SMA_{\text{long}} \quad \text{và} \quad SMA_{\text{short}, t-1} \ge SMA_{\text{long}, t-1}$$
- **Tín hiệu OBV**:
  $$\text{Mua khi: } OBV > MA(OBV) \quad \text{và} \quad OBV_{t-1} \le MA(OBV)_{t-1}$$
  $$\text{Bán khi: } OBV < MA(OBV) \quad \text{và} \quad OBV_{t-1} \ge MA(OBV)_{t-1}$$
- **Kết hợp OR**: Tín hiệu Mua kích hoạt khi SMA hoặc OBV báo mua; Tín hiệu Bán kích hoạt khi một trong hai báo bán. Tránh xung đột nếu cả hai chỉ báo phát tín hiệu trái chiều trong cùng phiên.
- **Thực thi lệnh $T+1$ & Chi phí**: Tín hiệu ở ngày $t$ sẽ vào/ra vị thế vào ngày $t+1$. Mỗi lần thay đổi vị thế đều bị trừ chi phí giao dịch (giả định chuẩn: `0.10%`).

### 3. Tối Ưu Hóa Danh Mục (Modern Portfolio Theory - MPT)
- Tối đa hóa tỷ lệ Sharpe trên tập Train với bài toán quy hoạch phi tuyến `scipy.optimize.minimize` (thuật toán SLSQP):
  $$\max_{w} \frac{\mu_p - r_f}{\sigma_p} \quad \text{s.t.} \quad \sum w_i = 1, \quad 0.05 \le w_i \le 0.40$$

---

## 📊 Bảng Tóm Tắt Kết Quả Kiểm Định (Out-Of-Sample 2022)

| Danh Mục Đầu Tư | Tổng Lợi Nhuận (Total Return) | Tỷ Suất Năm (Annual Return) | Biến Động Năm (Volatility) | Tỷ Lệ Sharpe | Mức Sụt Giảm Tối Đa (Max DD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Equal Weight Buy & Hold** (Benchmark) | **-42.27%** | -42.66% | 46.40% | -0.96 | **-60.52%** |
| **SMA + OBV Equal Weight** | **-8.63%** | -8.73% | 26.29% | -0.22 | **-31.85%** |
| **SMA + OBV MPT Weight** | **-3.21%** | -3.25% | 28.06% | **+0.02** | **-29.22%** |

> **Nhận Định Chính**: Chiến lược SMA + OBV phát huy hiệu quả quản trị rủi ro cực kỳ ấn tượng trong năm thị trường sụt giảm mạnh (2022), giúp bảo toàn vốn cho nhà đầu tư khi giảm thiểu mức lỗ từ `-42.27%` xuống chỉ còn `-3.21%` và cắt giảm hơn một nửa mức sụt giảm tối đa (Max Drawdown).

---

## 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ**: Python 3.9+
- **Giao diện Web**: [Streamlit](https://streamlit.io/)
- **Xử lý Dữ liệu**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Đồ thị Tương tác**: [Plotly](https://plotly.com/python/)
- **Chỉ báo Kỹ thuật**: [TA Library](https://technical-analysis-library-in-python.readthedocs.io/)
- **Tối ưu hóa**: [SciPy (SLSQP)](https://scipy.org/), [Hyperopt (TPE)](https://hyperopt.github.io/hyperopt/)

---

## 📜 Giấy Phép (License)

Dự án được phân phối dưới giấy phép **MIT License**. Bạn hoàn toàn có thể tự do sử dụng, chỉnh sửa và phát triển cho mục đích học tập cũng như nghiên cứu cá nhân.
