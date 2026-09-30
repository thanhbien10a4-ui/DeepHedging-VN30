# Checklist đồ án Deep Hedging

## 1. Dữ liệu và tiền xử lý
- [x] Thu thập dữ liệu VN30 spot + VN30F1M futures 2017–2026
- [ ] Tiền xử lý: log-return, rolling window, feature engineering  
  **Trạng thái:** phần log-return/feature engineering/alignment đã hoàn thành; vẫn cần chốt xử lý continuous futures/roll để coi là hoàn tất toàn bộ tiền xử lý.
- [ ] Xây dựng stock market simulator (phí, thuế, cơ chế thị trường phù hợp)  
  **Trạng thái:** đã có market-friction config, HNX/VSDC fees, roll-cost proxy và slippage stress; còn thiếu lịch sử tỷ lệ ký quỹ để tính PIT tax đầy đủ, broker fee, integer contract sizing và explicit contract-roll mechanics.
- [ ] Calibrate Heston model trên dữ liệu VN30  
  **Trạng thái:** đã có train-only physical-measure Heston estimation, validation-only selection/refinement và frozen generator; cần trình bày rõ đây không phải risk-neutral option calibration và còn mismatch volatility clustering.

## 2. Mô hình baseline
- [x] Cài đặt delta hedging + minimum-variance hedge ratio
- [x] Đánh giá PnL, Sharpe, Max Drawdown của baseline truyền thống

## 3. Deep Hedging
- [x] Xây dựng policy network RNN/LSTM cho chiến lược phòng hộ
- [x] Huấn luyện trên Heston synthetic data + convergence check
- [x] Fine-tune trên dữ liệu VN30 thực tế
- [x] So sánh PnL, CVaR, transaction cost với baseline truyền thống
- [x] Phân tích theo market regime: tăng / giảm / sideway
- [x] Ablation study: ảnh hưởng chi phí giao dịch và thanh khoản
- [x] Phân tích graceful degradation khi thị trường thay đổi

## 4. Báo cáo và ứng dụng
- [ ] Báo cáo đầy đủ, có biểu đồ hiệu suất tích lũy  
  **Trạng thái:** đã tạo cumulative-performance figure, drawdown figure và các bảng kết quả; chưa viết báo cáo hoàn chỉnh.
- [ ] Danh mục tài liệu tham khảo tối thiểu 15 tài liệu
- [x] Đưa cấu trúc mã nguồn tái lập lên GitHub
- [ ] Hoàn thiện `requirements.txt` với phiên bản môi trường cuối cùng
- [ ] Nêu ứng dụng thực tiễn cho nhà đầu tư sử dụng VN30 futures

## Danh sách mô hình dự kiến
| Mô hình | Vai trò | Trạng thái | Thông số ban đầu |
|---|---|---|---|
| Delta hedging | Baseline | Hoàn thành | hedge ratio / rebalance daily |
| Minimum-variance hedge | Baseline | Hoàn thành | OLS regression-based ratio |
| Deep Hedging LSTM | Đề xuất | Hoàn thành | hidden=128, lr=1e-4, CVaR alpha=0.05 |
| Deep Hedging GRU | Biến thể | Chưa triển khai | hidden=128, lr=1e-4 |
| DH + transaction cost penalty | Cải tiến | Hoàn thành ở dạng normalized turnover penalty; cost-aware OOS đánh giá riêng | lambda_tc=0.001 |

## Ghi chú phương pháp sau khi mở TEST
- REAL TEST 2025-01-02 → 2026-09-29 đã được mở ở bước 23 sau khi model được freeze.
- Không được retune model, feature, clipping threshold, objective hoặc simulator dựa trên TEST.
- Các bước 24–28 chỉ là post-freeze descriptive robustness/reporting analyses.
