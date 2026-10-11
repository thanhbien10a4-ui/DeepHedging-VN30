# Checklist đồ án Deep Hedging

Cập nhật sau đối chiếu artifacts ngày 11/10/2026. Các ô hoàn thành thực nghiệm xác nhận đã chạy và có bằng chứng; không xác nhận tái lập độc lập hoặc ưu thế thống kê.

## Quy ước trạng thái
- `[x]`: dữ liệu, mã nguồn hoặc tài liệu đã có trên GitHub, đủ xác nhận hoàn thành phần triển khai trong phạm vi ghi bên dưới.
- `[ ] — Đã triển khai, chờ kiểm chứng`: có code nhưng chưa có đủ đầu ra thực nghiệm trên GitHub để xác nhận hoàn thành việc chạy/đánh giá.
- `[ ] — Chưa hoàn thành`: còn thiếu sản phẩm theo yêu cầu.
- Thiếu đầu ra trên GitHub không có nghĩa là chưa chạy trên máy cá nhân. Các kết quả trong methodology là kết quả được báo cáo, đã đối chiếu với metrics/history và checksum checkpoint; chưa chạy lại độc lập.
- Không dùng tỷ lệ ô tick làm tỷ lệ khối lượng công việc hoặc tỷ lệ kết quả thực nghiệm đã xác minh.

## 1. Dữ liệu và tiền xử lý
- [x] Thu thập dữ liệu VN30 spot + VN30F1M futures 2017–2026  
  **Bằng chứng:** hai CSV trong `data/raw/`; dữ liệu được báo cáo đến 29/09/2026, không phải toàn bộ năm 2026.
- [x] Tiền xử lý: log-return, rolling window, feature engineering  
  **Phạm vi:** có pipeline bước 01–04 và 09 cho alignment, returns, rolling/lag features, statistical validation, rollover diagnostics và chronological splits. Metadata và reports đầu ra đã được lưu trên GitHub; các mảng dữ liệu đầy đủ có trong gói bàn giao. VN30F1M là continuous front-month series, không phải explicit historical contract-chain reconstruction.
- [x] Xây dựng stock market simulator (phí, thuế, cơ chế thị trường phù hợp)  
  **Phạm vi triển khai:** bước 30 và `configs/vn30f1m_market_friction.yaml`: danh mục VN30 10 tỷ VND, integer contract sizing, daily mark-to-market, margin giả định 50%, HNX/VSDC fees, broker fee 3.000 VND/HĐ/lần khớp, PIT theo project formula, slippage 1 tick và calendar roll proxy. Chưa mô phỏng intraday margin call, order-book fills hoặc hard cash-balance constraint. Tick này xác nhận xây dựng simulator; kết quả chạy cần kiểm chứng riêng.
- [x] Calibrate Heston model trên dữ liệu VN30  
  **Phạm vi triển khai:** có physical-measure historical estimation/calibration trên TRAIN, selection/refinement trên VALIDATION và frozen generator ở bước 05–14. Không phải risk-neutral option-surface calibration. Frozen metadata đã được bổ sung để đối chiếu tham số và validation gates; volatility-clustering mismatch và model strain phải giữ trong báo cáo.

## 2. Mô hình baseline
- [x] Cài đặt delta hedging + minimum-variance hedge ratio  
  **Bằng chứng:** bước 15 và các script OOS. Delta là hedge chỉ số 1:1; minimum variance có static TRAIN-estimated và causal rolling-60 variants.
- [x] Đánh giá PnL, Sharpe, Max Drawdown của baseline truyền thống  
  **Bằng chứng đã đối chiếu:** reports/CSV/figures được lưu trong `results/`; xem kết quả và giới hạn tại `docs/report/FINAL_REPORT.md`. Kiểm tra lịch sử xác nhận best epoch 11 (variance), 16 (CVaR), 2 (fine-tuning); early stopping không chứng minh tối ưu toàn cục.

- [x] Xây dựng policy network RNN/LSTM cho chiến lược phòng hộ  
  **Bằng chứng:** bước 17–18, 22: 7 state variables → LSTM một lớp, hidden=128 → linear head + sigmoid → hedge ratio trong [0, 1.5]. GRU chưa triển khai.
- [x] Huấn luyện trên Heston synthetic data + convergence check  
  **Bằng chứng đã đối chiếu:** reports/CSV/figures được lưu trong `results/`; xem kết quả và giới hạn tại `docs/report/FINAL_REPORT.md`. Kiểm tra lịch sử xác nhận best epoch 11 (variance), 16 (CVaR), 2 (fine-tuning); early stopping không chứng minh tối ưu toàn cục.

- [x] Fine-tune trên dữ liệu VN30 thực tế  
  **Bằng chứng đã đối chiếu:** reports/CSV/figures được lưu trong `results/`; xem kết quả và giới hạn tại `docs/report/FINAL_REPORT.md`. Kiểm tra lịch sử xác nhận best epoch 11 (variance), 16 (CVaR), 2 (fine-tuning); early stopping không chứng minh tối ưu toàn cục.

- [x] So sánh PnL, CVaR, transaction cost với baseline truyền thống  
  **Bằng chứng đã đối chiếu:** reports/CSV/figures được lưu trong `results/`; xem kết quả và giới hạn tại `docs/report/FINAL_REPORT.md`. Kiểm tra lịch sử xác nhận best epoch 11 (variance), 16 (CVaR), 2 (fine-tuning); early stopping không chứng minh tối ưu toàn cục.

- [x] Phân tích theo market regime: tăng / giảm / sideway  
  **Bằng chứng đã đối chiếu:** reports/CSV/figures được lưu trong `results/`; xem kết quả và giới hạn tại `docs/report/FINAL_REPORT.md`. Kiểm tra lịch sử xác nhận best epoch 11 (variance), 16 (CVaR), 2 (fine-tuning); early stopping không chứng minh tối ưu toàn cục.

- [ ] Ablation study: ảnh hưởng chi phí giao dịch và thanh khoản  
  **Đã triển khai một phần, chờ kiểm chứng:** bước 26 có cost ablation và slippage stress 0.5/1/2/4 ticks đại diện cho suy giảm thanh khoản. Cần kết quả chạy. Chưa có empirical liquidity analysis bằng bid–ask spread, order-book depth, participation rate hoặc hạn chế khớp lệnh; không gọi slippage stress là mô phỏng thanh khoản đầy đủ.
- [x] Phân tích graceful degradation khi thị trường thay đổi  
  **Bằng chứng đã đối chiếu:** reports/CSV/figures được lưu trong `results/`; xem kết quả và giới hạn tại `docs/report/FINAL_REPORT.md`. Kiểm tra lịch sử xác nhận best epoch 11 (variance), 16 (CVaR), 2 (fine-tuning); early stopping không chứng minh tối ưu toàn cục.

- [x] Báo cáo đầy đủ, có biểu đồ hiệu suất tích lũy  
  **Bằng chứng đã đối chiếu:** reports/CSV/figures được lưu trong `results/`; xem kết quả và giới hạn tại `docs/report/FINAL_REPORT.md`. Kiểm tra lịch sử xác nhận best epoch 11 (variance), 16 (CVaR), 2 (fine-tuning); early stopping không chứng minh tối ưu toàn cục.

- [x] Danh mục tài liệu tham khảo tối thiểu 15 tài liệu  
  **Bằng chứng:** `docs/literature/REFERENCES.md` có 22 mục đánh số. Phần mở đầu đã đồng bộ 22 nguồn; tick xác nhận số lượng danh mục, không xác nhận đã kiểm tra mọi DOI/link hoặc đã trích dẫn đầy đủ trong báo cáo.
- [x] Đưa cấu trúc mã nguồn tái lập lên GitHub  
  **Phạm vi:** raw data, source pipeline, configs và methodology đã có trên GitHub. Chưa kiểm chứng clean-environment reproduction; README và `docs/REPRODUCIBILITY.md` đã có hướng dẫn; `scripts/project.py` có audit/report/package/frozen evaluation. Còn cần kết quả tham chiếu và clean-environment reproduction để xác nhận tái lập.
- [x] Hoàn thiện `requirements.txt` với phiên bản môi trường cuối cùng  
  **Bằng chứng:** dependencies được pin và môi trường được ghi tại `docs/methodology/ENVIRONMENT.md`; có bước 29 capture environment. Chưa xác nhận cài đặt sạch và chạy lại thành công; `environment_snapshot.json` đã được bổ sung trên GitHub.
- [x] Nêu ứng dụng thực tiễn cho nhà đầu tư sử dụng VN30 futures  
  **Bằng chứng:** `docs/report/PRACTICAL_APPLICATION.md` trình bày đối tượng, quy trình paper trading, sizing, margin, rebalance/roll và điều kiện trước live deployment. Tick xác nhận phần trình bày ứng dụng, không phải đã vận hành giao dịch thật.

## Danh sách mô hình
| Mô hình | Vai trò | Trạng thái xác nhận trên GitHub |
|---|---|---|
| Delta hedge 1:1 | Baseline | Đã triển khai và có bảng kết quả tham chiếu |
| Static / causal rolling-60 minimum-variance hedge | Baseline | Đã triển khai và có bảng kết quả tham chiếu |
| Deep Hedging LSTM | Đề xuất | Đã có pretraining/fine-tuning histories, metrics và checksum; chưa tái chạy độc lập |
| Deep Hedging GRU | Biến thể dự kiến | Chưa triển khai; config không phải implementation |
| DH + transaction-cost penalty | Cải tiến | Đã triển khai normalized turnover penalty lambda_tc=0.001; không phải training trực tiếp theo toàn bộ phí/thuế/slippage thực tế |

## Bằng chứng cần bổ sung để đóng các mục thực nghiệm
- Training histories, convergence curves và sanity-check reports của bước 17–18.
- Fine-tuning report/history, best epoch và checkpoint metadata của bước 22.
- Baseline/OOS/cost/regime/ablation/stress/full-market metrics JSON, tables CSV và figures.
- Environment snapshot, hướng dẫn chạy theo thứ tự và kết quả kiểm chứng tái lập.
- Checkpoint có thể cung cấp qua release/artifact cùng checksum và hướng dẫn tải; không bắt buộc commit binary lớn vào Git.
- Không cần mở lại hoặc retune TEST chỉ để bổ sung đầu ra đã có.

## Ghi chú phương pháp sau khi mở TEST
- Theo methodology, REAL TEST 2025-01-02 → 2026-09-29 đã được mở ở bước 23 sau khi model được freeze.
- Không retune model, feature, clipping threshold, objective hoặc simulator dựa trên TEST.
- Các bước 24–30 là post-freeze descriptive robustness/reporting/economic-simulator analyses.
- 370 overlapping 60-day episodes không phải 370 quan sát độc lập; chưa có cơ sở suy ra xác suất model vượt baseline từ số episode này.
- Methodology ghi exploratory contamination caveat trước khi split policy được khóa; không tuyên bố TEST hoàn toàn pristine.
- Methodology và limitations: `docs/methodology/METHODOLOGY_AND_LIMITATIONS.md`.

## Bàn giao ngày 11/10/2026
- Đã bổ sung report phương pháp, ứng dụng, hướng dẫn pipeline, prompt nghiệm thu Antigravity, audit/report/package runner và test công cụ bàn giao.
- Test công cụ không thay thế inference/OOS reproduction. Các mục thực nghiệm đã được cập nhật sau khi đối chiếu artifacts.
- Không thay đổi mô hình, hyperparameters hoặc TEST selection.

## Nghiệm thu bằng chứng ngày 11/10/2026
- Đã đối chiếu 31 source Python với bản dự án: cùng nội dung sau chuẩn hóa newline CRLF/LF.
- Đã kiểm tra minimum của training histories và bảng OOS; checkpoint được xác nhận bằng checksum, chưa deserialization/inference lại.
- Kết quả và figures tham chiếu đã đưa vào GitHub. Các mô tả “chờ kiểm chứng” phía trên về tái lập môi trường sạch vẫn áp dụng cho reproduction, không cho việc tồn tại đầu ra.
- Mục ablation thanh khoản giữ chưa hoàn thành: đã có cost/slippage sensitivity, còn thiếu empirical liquidity analysis.
- Multiple seeds, statistical confidence intervals, GRU và clean-environment reproduction chưa hoàn thành, không tự thêm tick.
