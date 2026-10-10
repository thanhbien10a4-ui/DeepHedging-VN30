# Checklist đồ án Deep Hedging

Cập nhật sau rà soát GitHub ngày 10/10/2026.

## Quy ước trạng thái
- `[x]`: dữ liệu, mã nguồn hoặc tài liệu đã có trên GitHub, đủ xác nhận hoàn thành phần triển khai trong phạm vi ghi bên dưới.
- `[ ] — Đã triển khai, chờ kiểm chứng`: có code nhưng chưa có đủ đầu ra thực nghiệm trên GitHub để xác nhận hoàn thành việc chạy/đánh giá.
- `[ ] — Chưa hoàn thành`: còn thiếu sản phẩm theo yêu cầu.
- Thiếu đầu ra trên GitHub không có nghĩa là chưa chạy trên máy cá nhân. Các kết quả trong methodology là kết quả được báo cáo, chưa được đối chiếu độc lập với metrics/checkpoint.
- Không dùng tỷ lệ ô tick làm tỷ lệ khối lượng công việc hoặc tỷ lệ kết quả thực nghiệm đã xác minh.

## 1. Dữ liệu và tiền xử lý
- [x] Thu thập dữ liệu VN30 spot + VN30F1M futures 2017–2026  
  **Bằng chứng:** hai CSV trong `data/raw/`; dữ liệu được báo cáo đến 29/09/2026, không phải toàn bộ năm 2026.
- [x] Tiền xử lý: log-return, rolling window, feature engineering  
  **Phạm vi:** có pipeline bước 01–04 và 09 cho alignment, returns, rolling/lag features, statistical validation, rollover diagnostics và chronological splits. Dataset sinh ra không được lưu trên GitHub; số dòng đầu ra trong methodology cần đối chiếu khi kiểm chứng. VN30F1M là continuous front-month series, không phải explicit historical contract-chain reconstruction.
- [x] Xây dựng stock market simulator (phí, thuế, cơ chế thị trường phù hợp)  
  **Phạm vi triển khai:** bước 30 và `configs/vn30f1m_market_friction.yaml`: danh mục VN30 10 tỷ VND, integer contract sizing, daily mark-to-market, margin giả định 50%, HNX/VSDC fees, broker fee 3.000 VND/HĐ/lần khớp, PIT theo project formula, slippage 1 tick và calendar roll proxy. Chưa mô phỏng intraday margin call, order-book fills hoặc hard cash-balance constraint. Tick này xác nhận xây dựng simulator; kết quả chạy cần kiểm chứng riêng.
- [x] Calibrate Heston model trên dữ liệu VN30  
  **Phạm vi triển khai:** có physical-measure historical estimation/calibration trên TRAIN, selection/refinement trên VALIDATION và frozen generator ở bước 05–14. Không phải risk-neutral option-surface calibration. Các tham số cuối cùng và validation gates được báo cáo trong methodology cần đối chiếu với output; volatility-clustering mismatch và model strain phải giữ trong báo cáo.

## 2. Mô hình baseline
- [x] Cài đặt delta hedging + minimum-variance hedge ratio  
  **Bằng chứng:** bước 15 và các script OOS. Delta là hedge chỉ số 1:1; minimum variance có static TRAIN-estimated và causal rolling-60 variants.
- [ ] Đánh giá PnL, Sharpe, Max Drawdown của baseline truyền thống  
  **Đã triển khai, chờ kiểm chứng:** bước 15 có code tính metrics; cần bảng đầu ra. Bước 28 có reporting theo episode; không diễn giải descriptive Sharpe/drawdown từ episode chồng lấn như kết quả độc lập hoặc đường tài sản live liên tục.

## 3. Deep Hedging
- [x] Xây dựng policy network RNN/LSTM cho chiến lược phòng hộ  
  **Bằng chứng:** bước 17–18, 22: 7 state variables → LSTM một lớp, hidden=128 → linear head + sigmoid → hedge ratio trong [0, 1.5]. GRU chưa triển khai.
- [ ] Huấn luyện trên Heston synthetic data + convergence check  
  **Đã triển khai, chờ kiểm chứng:** bước 17–18 có training, early stopping, checkpoint selection, training-history export và sanity checks. Cần training-history CSV, training curves, report JSON và thông tin checkpoint. Early stopping/sanity checks không tự chứng minh convergence.
- [ ] Fine-tune trên dữ liệu VN30 thực tế  
  **Đã triển khai, chờ kiểm chứng:** bước 19–22 có chuẩn bị real sequences, domain-shift/zero-shot diagnostics và fine-tuning trên TRAIN, chọn checkpoint theo VALIDATION. Cần report, history và checkpoint metadata; kiểm tra `best_epoch` và `fine_tuning_improved_over_epoch0` để biết fine-tuning có cải thiện so với pretrained hay giữ epoch 0.
- [ ] So sánh PnL, CVaR, transaction cost với baseline truyền thống  
  **Đã triển khai, chờ kiểm chứng:** bước 23–24 và 30 có code so sánh. Cần metrics JSON và bảng CSV OOS/full-market simulator; chưa xác nhận độc lập các số liệu trong methodology.
- [ ] Phân tích theo market regime: tăng / giảm / sideway  
  **Đã triển khai, chờ kiểm chứng:** bước 25 có causal trailing-return classification với thresholds lấy từ TRAIN. Cần bảng/biểu đồ theo regime. “Sideways” là nhóm giữa theo phân vị TRAIN, không mặc nhiên chứng minh thị trường đi ngang theo một định nghĩa kinh tế khác.
- [ ] Ablation study: ảnh hưởng chi phí giao dịch và thanh khoản  
  **Đã triển khai một phần, chờ kiểm chứng:** bước 26 có cost ablation và slippage stress 0.5/1/2/4 ticks đại diện cho suy giảm thanh khoản. Cần kết quả chạy. Chưa có empirical liquidity analysis bằng bid–ask spread, order-book depth, participation rate hoặc hạn chế khớp lệnh; không gọi slippage stress là mô phỏng thanh khoản đầy đủ.
- [ ] Phân tích graceful degradation khi thị trường thay đổi  
  **Đã triển khai, chờ kiểm chứng:** bước 27 là open-loop forward-shock stress: giữ action từ state gốc rồi perturb forward returns. Cần bảng/biểu đồ stress. Không chứng minh policy thích ứng khi state bị stress vì action không được tính lại theo đường trạng thái stress. Failure mode downside được methodology báo cáo phải được đối chiếu và công bố.

## 4. Báo cáo và ứng dụng
- [ ] Báo cáo đầy đủ, có biểu đồ hiệu suất tích lũy  
  **Chưa hoàn thành:** `docs/report/` chưa có báo cáo đầy đủ. Có code bước 28 sinh cumulative/drawdown figures nhưng figures chưa có trên GitHub. Cumulative curve là mean within-episode 60-day path, không phải stitched chronological wealth curve.
- [x] Danh mục tài liệu tham khảo tối thiểu 15 tài liệu  
  **Bằng chứng:** `docs/literature/REFERENCES.md` có 22 mục đánh số. Phần mở đầu file còn ghi 20, cần đồng bộ; tick xác nhận số lượng danh mục, không xác nhận đã kiểm tra mọi DOI/link hoặc đã trích dẫn đầy đủ trong báo cáo.
- [x] Đưa cấu trúc mã nguồn tái lập lên GitHub  
  **Phạm vi:** raw data, source pipeline, configs và methodology đã có trên GitHub. Chưa kiểm chứng clean-environment reproduction; README còn mô tả trạng thái ban đầu và thiếu hướng dẫn chạy toàn pipeline. Cần cập nhật hướng dẫn và bộ kết quả tham chiếu để xác nhận khả năng tái lập.
- [x] Hoàn thiện `requirements.txt` với phiên bản môi trường cuối cùng  
  **Bằng chứng:** dependencies được pin và môi trường được ghi tại `docs/methodology/ENVIRONMENT.md`; có bước 29 capture environment. Chưa xác nhận cài đặt sạch và chạy lại thành công; `environment_snapshot.json` chưa có trên GitHub.
- [ ] Nêu ứng dụng thực tiễn cho nhà đầu tư sử dụng VN30 futures  
  **Chưa hoàn thành:** cần phần ứng dụng cụ thể, điều kiện sử dụng, quy trình hedge/rebalance và giới hạn về vốn, margin, rollover, execution cùng rủi ro ngoài mô hình.

## Danh sách mô hình
| Mô hình | Vai trò | Trạng thái xác nhận trên GitHub |
|---|---|---|
| Delta hedge 1:1 | Baseline | Đã triển khai; kết quả đánh giá chờ đối chiếu |
| Static / causal rolling-60 minimum-variance hedge | Baseline | Đã triển khai; kết quả đánh giá chờ đối chiếu |
| Deep Hedging LSTM | Đề xuất | Đã triển khai kiến trúc, pretraining và fine-tuning; convergence/checkpoint/metrics chờ kiểm chứng |
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
