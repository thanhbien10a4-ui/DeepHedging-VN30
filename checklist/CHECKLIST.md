# Checklist đồ án Deep Hedging

## 1. Dữ liệu và tiền xử lý
- [x] Thu thập dữ liệu VN30 spot + VN30F1M futures 2017–2026
- [x] Tiền xử lý: log-return, rolling window, feature engineering
- [ ] Xây dựng stock market simulator (phí, thuế, cơ chế thị trường phù hợp)
- [ ] Calibrate Heston model trên dữ liệu VN30

## 2. Mô hình baseline
- [ ] Cài đặt delta hedging + minimum-variance hedge ratio
- [ ] Đánh giá PnL, Sharpe, Max Drawdown của baseline truyền thống

## 3. Deep Hedging
- [ ] Xây dựng policy network RNN/LSTM cho chiến lược phòng hộ
- [ ] Huấn luyện trên Heston synthetic data + convergence check
- [ ] Fine-tune trên dữ liệu VN30 thực tế
- [ ] So sánh PnL, CVaR, transaction cost với baseline truyền thống
- [ ] Phân tích theo market regime: tăng / giảm / sideway
- [ ] Ablation study: ảnh hưởng chi phí giao dịch và thanh khoản
- [ ] Phân tích graceful degradation khi thị trường thay đổi

## 4. Báo cáo và ứng dụng
- [ ] Báo cáo đầy đủ, có biểu đồ hiệu suất tích lũy
- [ ] Danh mục tài liệu tham khảo tối thiểu 15 tài liệu
- [ ] Đưa cấu trúc mã nguồn tái lập lên GitHub
- [ ] Hoàn thiện `requirements.txt` với phiên bản môi trường cuối cùng
- [ ] Nêu ứng dụng thực tiễn cho nhà đầu tư sử dụng VN30 futures

## Danh sách mô hình dự kiến
| Mô hình | Vai trò | Thông số ban đầu |
|---|---|---|
| Delta hedging | Baseline | hedge ratio / rebalance daily |
| Minimum-variance hedge | Baseline | OLS regression-based ratio |
| Deep Hedging LSTM | Đề xuất | hidden=128, lr=1e-4, CVaR alpha=0.05 |
| Deep Hedging GRU | Biến thể | hidden=128, lr=1e-4 |
| DH + transaction cost penalty | Cải tiến | lambda_tc=0.001 |
