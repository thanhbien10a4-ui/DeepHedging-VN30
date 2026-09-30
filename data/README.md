# Data

## raw/
Chứa dữ liệu nguồn. Không chỉnh sửa trực tiếp.

### vn30_1d.csv
- Asset: VN30 spot index
- Frequency: Daily
- Rows: 2,279
- Date range: 2017-08-10 → 2026-09-29
- Columns: `time, open, high, low, close, volume`

### vn30f1m_1d.csv
- Asset: VN30F1M futures
- Frequency: Daily
- Rows: 2,281
- Date range: 2017-08-10 → 2026-09-29
- Columns: `time, open, high, low, close, volume`

## processed/
Dữ liệu đã được tạo lại bằng pipeline preprocessing. Không sửa thủ công.

### Vấn đề cần xử lý
VN30F1M có 2 ngày chưa có counterpart trong VN30 spot: `2018-01-23` và `2018-01-24`. Quyết định xử lý phải được ghi rõ trong preprocessing và báo cáo.
