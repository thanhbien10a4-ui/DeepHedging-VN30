# DeepHedging-VN30

Ứng dụng Deep Learning (học sâu) cho phòng hộ danh mục tương đương VN30 bằng short VN30F1M. Heston+basis synthetic pretraining, policy LSTM hidden=128, 7 state variables, fine-tune real TRAIN. Không phải tái lập nguyên bài toán payoff quyền chọn gốc.

## Trạng thái
Source pipeline 01–30, baselines, simulator integer-contract, báo cáo phương pháp và ứng dụng đã có. Metrics, CSV, biểu đồ và báo cáo kết quả đã có trên GitHub; checkpoint được xác nhận qua checksum nhưng không lưu trực tiếp trong repo. Các kết quả chưa được tái chạy độc lập trong môi trường sạch; GRU chưa triển khai. Xem [checklist](checklist/CHECKLIST.md).

## Cài đặt môi trường ảo
Requirements pin môi trường Windows CPU được dự án báo cáo; chưa chứng nhận cài sạch thành công. GPU cần lock riêng, không tự upgrade để thay đổi kết quả freeze.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Kiểm tra và đóng gói đầu ra hiện có
```powershell
python scripts/project.py audit
python scripts/project.py report
python scripts/project.py package --include-models
```
Audit exit 1 là thiếu/lỗi bằng chứng; exit 0 chỉ xác nhận artifact checks và vẫn cần người rà soát. Báo cáo Markdown/HTML ở results/report, ZIP+SHA-256 manifest ở dist. Default package không chứa model, raw CSV hay synthetic arrays; --include-models thêm checkpoint. --allow-incomplete chỉ tạo diagnostic bundle, không phải nghiệm thu.

## Đánh giá model đã freeze
```powershell
python scripts/project.py evaluate
```
Chạy 24–28, 30, capture environment 29; không train/tune/chạy lại 23. Yêu cầu checkpoint, splits, final TEST sequences có sẵn; sao lưu output trước chạy. Kiểm tra hashes để phát hiện đầu vào bị thay đổi.

## Tài liệu
- [Pipeline và tái lập](docs/REPRODUCIBILITY.md)
- [Báo cáo phương pháp](docs/report/FINAL_REPORT.md)
- [Ứng dụng thực tiễn](docs/report/PRACTICAL_APPLICATION.md)
- [Antigravity nghiệm thu](docs/ANTIGRAVITY_ACCEPTANCE.md)
- [Methodology/limitations](docs/methodology/METHODOLOGY_AND_LIMITATIONS.md)
- [22 nguồn tham khảo](docs/literature/REFERENCES.md)

## Giới hạn
Continuous futures, roll-calendar proxy, margin giả định 50%, không intraday fills/margin call/hard cash constraint. Turnover penalty không phải full official cost. Episode chồng lấn không độc lập; cumulative curve là mean within-episode. TEST đã mở và có exploratory contamination caveat; không retune theo TEST. Historical fee config không phải biểu phí live hiện hành. Tái lập từ đầu ở thư mục riêng để không ghi đè kết quả cũ.
