# Checklist đồ án Deep Hedging

## 1. Dữ liệu và tiền xử lý
- [x] Thu thập dữ liệu VN30 spot + VN30F1M futures 2017–2026
- [x] Tiền xử lý: log-return, rolling window, feature engineering  
  **Phạm vi hoàn thành:** alignment, log-return, rolling/lag features và model-ready dataset đã hoàn thành. VN30F1M được xử lý dưới dạng continuous front-month series; rollover được document và xử lý bằng calendar roll proxy trong economic evaluation, không tuyên bố explicit historical contract-chain reconstruction.
- [x] Xây dựng stock market simulator (phí, thuế, cơ chế thị trường phù hợp)  
  **Phạm vi hoàn thành:** danh mục VN30 10 tỷ VND, integer contract sizing, daily mark-to-market, margin giả định 1:2 (50%), HNX/VSDC fees, broker fee 3.000 VND/HĐ/lần khớp, PIT tax theo project formula, slippage 1 tick và calendar roll proxy. Các giới hạn được đóng gói tại `docs/methodology/METHODOLOGY_AND_LIMITATIONS.md`.
- [x] Calibrate Heston model trên dữ liệu VN30  
  **Phạm vi hoàn thành:** physical-measure historical Heston estimation/calibration trên VN30 TRAIN, validation-only selection/refinement và frozen synthetic generator. Đây không phải risk-neutral option-surface calibration; standard-Heston volatility-clustering mismatch được ghi rõ trong methodology/limitations.

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
  **Trạng thái:** đã tạo cumulative-performance figure, drawdown figure, regime/cost/stress figures và các bảng kết quả; chưa viết báo cáo hoàn chỉnh.
- [x] Danh mục tài liệu tham khảo tối thiểu 15 tài liệu  
  **Trạng thái:** `docs/literature/REFERENCES.md` hiện có 22 nguồn gồm Deep Hedging, CVaR, Heston, futures hedging, volatility và nguồn chính thức VN30 futures.
- [x] Đưa cấu trúc mã nguồn tái lập lên GitHub
- [x] Hoàn thiện `requirements.txt` với phiên bản môi trường cuối cùng
- [ ] Nêu ứng dụng thực tiễn cho nhà đầu tư sử dụng VN30 futures

## Danh sách mô hình dự kiến
| Mô hình | Vai trò | Trạng thái | Thông số ban đầu |
|---|---|---|---|
| Delta hedging | Baseline | Hoàn thành | hedge ratio / rebalance daily |
| Minimum-variance hedge | Baseline | Hoàn thành | OLS regression-based ratio |
| Deep Hedging LSTM | Đề xuất | Hoàn thành | hidden=128, lr=1e-4, CVaR alpha=0.05 |
| Deep Hedging GRU | Biến thể | Chưa triển khai | hidden=128, lr=1e-4 |
| DH + transaction cost penalty | Cải tiến | Hoàn thành ở dạng normalized turnover penalty; cost-aware/full-market OOS đánh giá riêng | lambda_tc=0.001 |

## Ghi chú phương pháp sau khi mở TEST
- REAL TEST 2025-01-02 → 2026-09-29 đã được mở ở bước 23 sau khi model được freeze.
- Không được retune model, feature, clipping threshold, objective hoặc simulator dựa trên TEST.
- Các bước 24–30 chỉ là post-freeze descriptive robustness/reporting/economic-simulator analyses.
- Methodology và limitations chính thức: `docs/methodology/METHODOLOGY_AND_LIMITATIONS.md`.
