# Deep Hedging for VN30 Futures

Đồ án nghiên cứu **Deep Hedging phòng hộ rủi ro trên thị trường chứng khoán Việt Nam**, sử dụng VN30 spot và hợp đồng tương lai VN30F1M.

## Trạng thái hiện tại

Repository được khởi tạo từ hai bộ dữ liệu chính và checklist đồ án.

| Dataset | Khung thời gian | Số dòng | Từ ngày | Đến ngày | Missing | Duplicate date |
|---|---:|---:|---|---|---:|---:|
| VN30 spot | 1D | 2,279 | 2017-08-10 | 2026-09-29 | 0 | 0 |
| VN30F1M futures | 1D | 2,281 | 2017-08-10 | 2026-09-29 | 0 | 0 |

**Lưu ý dữ liệu:** VN30F1M hiện có thêm 2 ngày `2018-01-23` và `2018-01-24` mà VN30 spot không có. Cần xử lý việc đồng bộ lịch giao dịch trong bước preprocessing, không tự động xóa dữ liệu gốc.

## Mục tiêu đồ án

1. Thu thập và kiểm tra dữ liệu VN30 spot + VN30F1M futures giai đoạn 2017–2026.
2. Tiền xử lý: log-return, rolling windows, feature engineering và đồng bộ chuỗi thời gian.
3. Xây dựng stock market simulator có chi phí giao dịch và các ràng buộc thị trường Việt Nam phù hợp với bài toán.
4. Calibrate mô hình Heston trên dữ liệu VN30.
5. Xây dựng baseline hedging truyền thống.
6. Xây dựng Deep Hedging với RNN/LSTM và các biến thể.
7. Huấn luyện trên synthetic data, sau đó fine-tune/evaluate trên dữ liệu thực tế.
8. So sánh PnL, CVaR, Sharpe, Max Drawdown và transaction cost.
9. Phân tích theo market regime, ablation study và graceful degradation.
10. Bảo đảm mã nguồn và thí nghiệm có thể tái lập.

## Baseline và mô hình dự kiến

- Delta hedging
- Minimum-variance hedge ratio
- Deep Hedging LSTM
- Deep Hedging GRU
- Deep Hedging + transaction-cost penalty

## Cấu trúc repository

```text
DeepHedging-VN30/
├── data/
│   ├── raw/                  # Dữ liệu gốc, không sửa trực tiếp
│   └── processed/            # Dữ liệu sinh ra sau preprocessing
├── checklist/                # Checklist yêu cầu của giảng viên
├── notebooks/                # Notebook khám phá, demo và thí nghiệm
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── simulator/
│   ├── models/
│   ├── hedging/
│   ├── evaluation/
│   └── utils/
├── configs/                  # Hyperparameters / experiment configs
├── experiments/              # Lịch sử thí nghiệm
├── results/
│   ├── figures/
│   ├── tables/
│   └── metrics/
├── docs/
│   ├── methodology/
│   ├── literature/
│   └── report/
└── tests/
```

## Quy tắc quản lý dữ liệu

- `data/raw/` là **immutable**: không sửa trực tiếp file nguồn.
- Mọi bước làm sạch/chuyển đổi phải được thực hiện bằng code và ghi kết quả vào `data/processed/`.
- Không commit `.venv`, cache, checkpoint lớn hoặc dữ liệu tạm.
- Mỗi thí nghiệm quan trọng phải lưu config và metrics để có thể tái lập.

## Checklist chính

Chi tiết đầy đủ xem tại [`checklist/CHECKLIST.md`](checklist/CHECKLIST.md) và file CSV gốc trong cùng thư mục.

## Reproducibility

Môi trường sẽ được khóa dần trong quá trình triển khai. `requirements.txt` hiện là bộ dependency nền tảng; phiên bản deep-learning framework cần được chốt theo môi trường chạy chính thức của đồ án trước khi đóng băng phiên bản cuối.
