# DeepHedging-VN30 — Tóm tắt dự án dùng cho CV

**Thời gian:** 2026  
**Lĩnh vực:** Deep Learning, Time Series, Quantitative Finance  
**Mã nguồn:** https://github.com/thanhbien10a4-ui/DeepHedging-VN30

## Mô tả

Xây dựng quy trình Deep Hedging cho danh mục có mức độ phơi nhiễm tương đương chỉ số VN30, sử dụng hợp đồng tương lai VN30F1M để phòng hộ. Mô hình học tỷ lệ phòng hộ theo chuỗi trạng thái thị trường, với mục tiêu kiểm soát rủi ro đuôi và turnover. Đây là ứng dụng cho phòng hộ danh mục chỉ số, không phải tái lập nguyên bài toán payoff quyền chọn trong paper Deep Hedging gốc.

## Công việc đã triển khai

- Xây dựng pipeline dữ liệu ngày cho VN30 spot và continuous front-month VN30F1M: căn chỉnh chuỗi, tính log-return và rolling features, kiểm định thống kê, chẩn đoán rollover/outlier, và chia dữ liệu theo thời gian.
- Ước lượng Heston physical-measure và basis AR(1) trên TRAIN; chọn/tinh chỉnh trên VALIDATION; sinh dữ liệu synthetic kết hợp spot–futures và kiểm tra chênh lệch phân phối giữa synthetic với dữ liệu thật.
- Huấn luyện policy LSTM một lớp (hidden size 128, 7 biến trạng thái, hedge ratio trong [0, 1.5]) trên dữ liệu synthetic; objective là path-CVaR ở 5% đuôi với normalized turnover penalty. Fine-tune bằng dữ liệu VN30/VN30F1M thực.
- So sánh với các baseline unhedged, hedge 1:1, static minimum-variance và causal rolling-60 minimum-variance.
- Xây dựng simulator kinh tế với sizing hợp đồng nguyên, phí/thuế theo giả định của nghiên cứu, slippage, margin và calendar-roll proxy; đánh giá theo PnL, CVaR, turnover, chi phí và market regime.
- Tổ chức mã nguồn, cấu hình, báo cáo phương pháp, artifacts kết quả và hướng dẫn chạy trên GitHub.

## Kết quả có thể nêu

Trong đánh giá simulator đã freeze trên giai đoạn TEST 02/01/2025–29/09/2026, sử dụng 370 cửa sổ 60 phiên chồng lấn, policy LSTM fine-tuned đạt **net CVaR loss thấp hơn 9,22% so với hedge 1:1** theo cấu hình giả lập danh mục exposure 10 tỷ VND và chi phí đã khai báo. Chênh lệch net mean PnL so với hedge 1:1 là khoảng **+27,3 triệu VND** theo cùng bộ episode.

Đây là kết quả backtest mô phỏng mang tính mô tả, chưa có kiểm định ý nghĩa thống kê hay khoảng tin cậy. Các episode chồng lấn nên không phải 370 quan sát độc lập; kết quả không chứng minh hiệu quả giao dịch live hoặc ưu thế chắc chắn ngoài mẫu. Đánh giá CVaR trên TEST đã mở và không được dùng để retune mô hình.

## Nội dung CV đề xuất — tiếng Việt

**DeepHedging-VN30 | Dự án Deep Learning cho phòng hộ VN30 Futures**  
- Xây dựng pipeline dữ liệu và mô phỏng spot–futures; ước lượng Heston và basis AR(1) để tạo dữ liệu synthetic cho tiền huấn luyện.  
- Huấn luyện policy LSTM tối ưu path-CVaR (đuôi 5%) có turnover penalty, sau đó fine-tune trên dữ liệu VN30/VN30F1M; đánh giá với hedge 1:1 và minimum-variance baselines.  
- Phát triển simulator chi phí giao dịch với sizing hợp đồng nguyên, margin, slippage và roll proxy; trong backtest mô phỏng đã freeze, net CVaR loss thấp hơn 9,22% so với hedge 1:1 trên 370 cửa sổ 60 phiên chồng lấn.

## English CV wording

**DeepHedging-VN30 | Deep Learning for VN30 Futures Hedging**  
- Built a daily VN30 spot/front-month futures data pipeline, including feature engineering, rollover diagnostics, chronological splits, and a Heston–AR(1) synthetic market generator.  
- Trained a 7-state, single-layer LSTM hedging policy with a 5% tail-CVaR objective and turnover penalty, then fine-tuned it on real VN30/VN30F1M data; benchmarked against 1:1 and minimum-variance hedges.  
- Implemented an integer-contract market simulator with transaction-cost, margin, slippage, and calendar-roll assumptions; the frozen backtest reported 9.22% lower net CVaR loss than the 1:1 hedge across 370 overlapping 60-day windows.

## Những điều không nên ghi quá mức

- Không gọi 9,22% là lợi nhuận tăng thêm; đây là mức giảm CVaR loss so với một baseline trong simulator.
- Không nói mô hình đã được chứng minh vượt trội có ý nghĩa thống kê, chạy giao dịch thật, hoặc tái lập độc lập trong môi trường sạch.
- Không gọi slippage stress là phân tích thanh khoản thực nghiệm; repo chưa có order-book depth, bid–ask spread hoặc execution-fill data.
- Không ghi GRU là mô hình đã triển khai; repository hiện có cấu hình GRU nhưng chưa có GRU implementation.
