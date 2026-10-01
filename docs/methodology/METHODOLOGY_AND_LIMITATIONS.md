# Methodology and Limitations — DeepHedging-VN30

Tài liệu này đóng gói phương pháp chính thức của đồ án và phạm vi các kết luận được phép đưa vào báo cáo. Mục tiêu là phân biệt rõ **những gì đã hoàn thành trong phạm vi đồ án** với **những gì không nên tuyên bố vượt quá dữ liệu và thiết kế hiện có**.

---

## 1. Dữ liệu và tiền xử lý

### 1.1 Dữ liệu gốc

Đồ án sử dụng dữ liệu ngày của:

- VN30 spot/index.
- VN30F1M futures/front-month continuous series.
- Giai đoạn tổng thể: 2017-08-10 đến 2026-09-29.

Sau bước validation/alignment:

- VN30: 2,279 dòng.
- VN30F1M: 2,281 dòng.
- Ngày giao dịch chung: 2,279.
- Hai ngày chỉ có futures: 2018-01-23 và 2018-01-24.

Hai ngày lệch này không được tự nội suy để tạo spot price giả; modeling dataset chỉ sử dụng lịch giao dịch đã align.

### 1.2 Feature engineering

Feature engineering được thực hiện theo nguyên tắc causal: state tại thời điểm `t` chỉ sử dụng thông tin đã quan sát được tại hoặc trước `t` theo giả định ra quyết định tại close.

Pipeline tạo 28 feature gồm các nhóm:

- spot/futures log-return và lagged return;
- basis và basis velocity;
- momentum/trend;
- range/volatility và realized volatility;
- volume/liquidity-derived features;
- calendar/time-related state.

Lookback lớn nhất là 60 phiên. Vì vậy:

- input aligned: 2,279 dòng;
- warm-up bị loại: 60 dòng;
- model-ready: 2,219 dòng;
- model-ready bắt đầu từ 2017-11-03.

Scaler không được fit trên toàn bộ dữ liệu. Với deep-learning pipeline, scaler được fit trên synthetic TRAIN và tái sử dụng ở các bước transfer; không fit lại trên real validation/test.

### 1.3 Continuous futures và rollover

VN30F1M hiện có trong đồ án là **continuous front-month series**, không có contract ID của từng hợp đồng đáo hạn. Vì vậy đồ án không tuyên bố đã tái dựng hoàn hảo chain hợp đồng lịch sử.

Quy ước được khóa cho phạm vi đồ án:

- continuous VN30F1M được dùng làm price process cho modeling;
- ngày roll được xác định bằng calendar expiry proxy: thứ Năm thứ ba của tháng, hoặc phiên quan sát gần nhất trước đó nếu ngày mục tiêu không có dữ liệu;
- khi đánh giá market frictions, rollover được mô hình hóa bằng việc đóng vị thế hợp đồng cũ và mở lại cùng quy mô ở hợp đồng kế tiếp;
- roll proxy làm tăng transaction quantity/cost nhưng **không sửa lại chuỗi giá continuous** vì dữ liệu không chứa giá hai contract riêng tại thời điểm roll.

Với phạm vi này, preprocessing được xem là hoàn thành cho mục tiêu **model-ready continuous-futures hedging study**, nhưng không được gọi là explicit contract-chain reconstruction hoặc back-adjusted historical roll reconstruction.

---

## 2. Chia TRAIN / VALIDATION / TEST và kiểm soát leakage

Real data được chia chronological, không shuffle:

- TRAIN: 2017-08-10 đến 2022-12-30, 1,350 dòng.
- VALIDATION: 2023-01-03 đến 2024-12-31, 499 dòng.
- TEST: 2025-01-02 đến 2026-09-29, 430 dòng.

Quy tắc sử dụng:

- TRAIN: estimate/calibrate parameters và fine-tune model.
- VALIDATION: model/simulator selection và early stopping trước khi freeze.
- TEST: final OOS evaluation sau khi model đã freeze.

Real TEST được intentionally opened ở step 23. Sau step 23:

- không được retune architecture;
- không đổi feature schema;
- không đổi clipping threshold/objective để tối ưu TEST;
- không được đổi model final dựa trên việc pretrained model tình cờ có metric TEST tốt hơn ở một số tiêu chí.

Các bước 24–30 là post-freeze descriptive evaluation/robustness/economic-simulator analyses.

### Leakage caveat

Trước khi chronological split được khóa, một bước exploratory cũ đã từng nhìn aggregate properties của giai đoạn 2025–2026 trong synthetic-vs-real diagnostics. Vì vậy TEST không hoàn toàn pristine theo tiêu chuẩn nghiêm ngặt nhất. Tuy nhiên từ thời điểm split policy được khóa, toàn bộ model selection/fine-tuning chính đều tuân thủ TRAIN/VALIDATION và TEST chỉ được mở sau freeze.

Trong báo cáo nên mô tả đây là **exploratory contamination caveat**, không che giấu và cũng không diễn giải quá mức thành direct model tuning on TEST.

---

## 3. Heston calibration / historical parameter estimation

### 3.1 Phạm vi calibration

Classical Heston calibration trong derivative pricing thường là risk-neutral calibration trên option prices/implied-volatility surface. Đồ án không có lịch sử VN30 option surface phù hợp để thực hiện loại calibration đó.

Do đó, thuật ngữ đúng trong đồ án là:

> **Physical-measure historical Heston parameter estimation/calibration from VN30 underlying data, used to construct a standard-Heston synthetic training baseline.**

Không được viết rằng đồ án đã thực hiện risk-neutral Heston calibration từ option surface.

### 3.2 Variance proxy và estimation

Daily range information của VN30 được dùng để xây dựng Garman–Klass variance proxy. Nhiều smoothing window được so sánh (1, 5, 10, 20 ngày) để quan sát độ nhạy của các tham số CIR/Heston.

Một vấn đề quan trọng được phát hiện: rolling variance proxy tạo overlap, làm lag-1 variance correlation tăng cơ học khi window lớn hơn. Vì vậy không chọn estimator chỉ vì Feller condition đẹp hơn.

Pipeline cuối cùng:

1. Estimate structural Heston candidates trên real TRAIN only.
2. Estimate basis process trên TRAIN.
3. Dùng real VALIDATION để chọn/refine simulation adjustment.
4. Freeze simulator trước khi mở real TEST.
5. Generate final synthetic sets bằng frozen configuration.

### 3.3 Synthetic generator

Synthetic market kết hợp:

- Heston spot process, full-truncation Euler discretization;
- stochastic variance;
- AR(1) basis process;
- futures được dựng theo `F_t = S_t * (1 + b_t)`;
- 60 trading days/path;
- 20,000 synthetic TRAIN paths;
- 4,000 synthetic VALIDATION paths;
- 4,000 synthetic TEST paths.

Final frozen generator pass 6/7 core validation gates.

Known mismatch quan trọng:

- real-validation squared-return ACF1 target khoảng `-0.0187`;
- frozen synthetic squared-return ACF1 khoảng `0.1429`.

Nói cách khác, standard Heston baseline trong đồ án **overstates short-lag volatility clustering** so với validation period.

### 3.4 Heston limitations cần ghi trong báo cáo

- Không có option surface => không phải risk-neutral calibration.
- Variance là latent; historical variance proxy chỉ là approximation.
- Final validation refinement đẩy effective rho tới boundary mạnh (`rho_effective = -0.95`), cho thấy model strain.
- Standard Heston không tái tạo hoàn hảo jumps, heavy tails, expiry effects và regime shifts.
- Feller-condition violation ở một số candidate không tự động đồng nghĩa simulator vô hiệu; discretization và empirical fit vẫn phải được xét riêng.
- Frozen Heston được dùng như **structural synthetic pretraining baseline**, không được trình bày như true data-generating process của VN30.

Với định nghĩa trên, mục “Calibrate Heston model trên dữ liệu VN30” được xem là hoàn thành trong phạm vi historical/physical-measure calibration của đồ án.

---

## 4. Basis process

Basis được định nghĩa:

`b_t = (F_t - S_t) / S_t`.

AR(1) train-only baseline sau khi loại các transition gắn với likely-roll proxy có xấp xỉ:

- `phi ≈ 0.7637`;
- long-run mean khoảng `-0.00283`;
- residual std khoảng `0.00566`;
- half-life khoảng `2.57` trading days.

Đây là empirical baseline, không phải no-arbitrage cost-of-carry model đầy đủ. Nó không explicit modeling dividend yield, funding rate, days-to-expiry hay delivery/settlement basis convergence.

---

## 5. Deep Hedging methodology

### 5.1 Episode design

Deep Hedging sử dụng 60-day episodes. Action `h_t` áp dụng cho interval kế tiếp.

Feature schema final gồm 7 state variables:

1. lag spot log return;
2. lag futures log return;
3. current basis;
4. volatility state;
5. relative log spot level;
6. relative log futures level;
7. time remaining.

Synthetic feature dùng latent Heston variance. Khi transfer sang real data, volatility state được thay bằng causal same-day annualized Garman–Klass sqrt-volatility proxy. Đây là semantic/domain mismatch đã được chẩn đoán và phải ghi rõ.

### 5.2 Policy network và objective

Final architecture:

- 1-layer LSTM;
- hidden size 128;
- hedge output bounded trong `[0, 1.5]`;
- CVaR worst-tail fraction 5%;
- normalized turnover penalty `lambda_tc = 0.001` trong training objective.

Training flow:

1. pretrain trên frozen Heston+basis synthetic data;
2. evaluate zero-shot transfer trên real TRAIN/VALIDATION;
3. clip neural-network inputs ở ±5z cho bounded-input fine-tuning policy;
4. fine-tune trên real TRAIN;
5. early-stop/select bằng real VALIDATION;
6. freeze model;
7. open real TEST một lần để final OOS evaluation.

Fine-tuned model là proposed final model, kể cả khi synthetic-pretrained comparator có một vài TEST metrics tốt hơn. Không được đổi model sau khi nhìn TEST.

---

## 6. Traditional baselines

Đồ án dùng:

- unhedged;
- delta 1:1 hedge;
- static minimum-variance hedge estimated trên real TRAIN;
- causal rolling-60 minimum-variance hedge dùng chỉ lịch sử trước mỗi quyết định.

Đối với long VN30 spot exposure, futures hedge là short VN30F1M.

Static minimum-variance ratio trên TRAIN xấp xỉ `0.760765`.

---

## 7. Full VN30F1M market simulator

### 7.1 Portfolio và position sizing

Project simulator step 30 sử dụng:

- initial VN30 portfolio exposure: **10,000,000,000 VND**;
- contract multiplier: **100,000 VND/index point**;
- futures hedge direction: short;
- contract quantity: integer non-negative;
- rounding: nearest integer, half-up.

Target contract count:

`N_t = round_half_up(h_t * V_t / (F_t * multiplier))`.

Trong đó `V_t` là spot portfolio value theo diễn biến VN30 từ episode start.

### 7.2 Margin assumption

Giả định project được khóa là **margin 1:2**, tức:

- initial margin rate = `50%`;
- 1 đơn vị margin hỗ trợ 2 đơn vị futures notional.

Đây là **project simplification**, không phải historical VN30F1M exchange margin schedule.

10 tỷ VND được hiểu là spot portfolio exposure cần hedge, **không phải hard combined cash constraint** bao gồm đồng thời spot portfolio và futures margin account.

### 7.3 Daily mark-to-market

Futures PnL được tính daily mark-to-market:

`PnL_fut,t = -N_t * multiplier * (F_{t+1} - F_t)`.

Spot portfolio economic PnL được tính từ số units VN30-equivalent cố định trong episode.

### 7.4 Transaction costs

Simulator final gồm:

- HNX trading fee: 2,700 VND/contract/match;
- broker commission: 3,000 VND/contract/match;
- VSDC pre-KRX position-management fee theo config;
- VSDC post-KRX clearing fee theo config;
- default slippage: 1 tick;
- 1 tick = 0.1 index point = 10,000 VND/contract với multiplier 100,000;
- PIT tax theo formula của project;
- rollover matched quantity qua calendar proxy.

### 7.5 PIT tax assumption

PIT được tính theo project formula:

`tax = futures_execution_price * multiplier * traded_contracts * initial_margin_rate / 2 * 0.1%`.

Với project margin assumption 1:2:

`initial_margin_rate = 0.5`.

Đây là giả định thống nhất cho simulator, không đại diện cho full historical margin schedule 2017–2026.

### 7.6 Rollover

Tại expiry proxy date:

- close old contract;
- open next contract;
- transaction quantity tăng thêm `2 * current_contracts`;
- fees/tax/slippage được tính trên matched quantity;
- continuous front-month price series không bị chỉnh sửa.

### 7.7 Simulator result sanity check

Ở real TEST 2025–2026, 370 overlapping 60-day episodes:

- final real-finetuned LSTM net CVaR khoảng 121.9 triệu VND;
- delta 1:1 khoảng 134.3 triệu VND;
- rolling min-var khoảng 192.9 triệu VND;
- static min-var khoảng 267.8 triệu VND;
- final LSTM giảm CVaR khoảng 9.22% so với delta;
- mean integer contracts của final LSTM khoảng 57.8;
- realized hedge ratio trung bình khoảng 0.9718;
- max observed margin requirement khoảng 6.29 tỷ VND.

Các con số này là post-freeze descriptive OOS economic-simulator results, không được dùng để retune model.

### 7.8 Simulator limitations

- Continuous VN30F1M không có contract IDs => roll spread thực tế không thể tái dựng chính xác.
- 50% margin là giả định project, không phải historical schedule.
- Margin-asset management fee dạng account/month không được allocate vào từng hedge trade trong core simulator.
- Daily data không mô phỏng intraday margin call, intraday liquidity, order-book impact hay limit-order fills.
- Slippage 1 tick là project assumption/base case; các step 24–26 đã chạy sensitivity/stress riêng.
- Integer contracts được modeling, nhưng không có hard cash-balance constraint kết hợp spot + margin + variation margin.

Với các giả định và giới hạn được công bố rõ ràng, mục “Xây dựng stock market simulator” được xem là hoàn thành trong phạm vi đồ án.

---

## 8. OOS robustness và graceful degradation

Post-freeze analyses gồm:

- market regime: bear / sideways / bull bằng causal trailing-60 classification với thresholds lấy từ TRAIN;
- transaction-cost and slippage sensitivity;
- liquidity stress tới 4 ticks;
- graceful degradation under volatility/noise/downside shocks;
- cumulative performance và drawdown reporting.

Một failure mode phải được giữ nguyên trong báo cáo:

- dưới asymmetric `downside_x1p50` open-loop stress, final LSTM có CVaR tệ hơn delta 1:1 khoảng 40.56%.

Không được che failure mode này. Nó cho thấy policy không robust hoàn toàn trước asymmetric downside distribution shift.

---

## 9. Những claim được phép và không được phép

### Được phép viết

- Đồ án xây dựng một Deep Hedging LSTM cho VN30/VN30F1M với synthetic Heston pretraining và real-data fine-tuning.
- Heston được historically calibrated/estimated dưới physical measure từ VN30 underlying data.
- Full market simulator có integer contract sizing, margin assumption, fees, tax, slippage, daily MTM và roll-cost proxy.
- Final frozen LSTM cải thiện OOS CVaR so với các traditional baselines trong final simulator under stated assumptions.
- Robustness không đồng đều; có identifiable failure modes.

### Không được viết

- Heston đã được risk-neutral calibrated từ VN30 option surface.
- Continuous futures series tái dựng chính xác mọi historical contract roll.
- 50% margin là historical exchange margin của toàn bộ 2017–2026.
- TEST hoàn toàn pristine theo chuẩn nghiêm ngặt nhất.
- Overlapping 60-day episodes là independent observations.
- Final model luôn tốt hơn delta trong mọi regime/stress.
- Kết quả simulator là live-trading guarantee hoặc executable PnL chính xác tới từng hợp đồng lịch sử.

---

## 10. Checklist closure rationale

Ba mục sau được đóng trong **phạm vi project scope đã công bố**:

1. **Tiền xử lý: log-return, rolling window, feature engineering — COMPLETE**  
   Model-ready pipeline hoàn thành; continuous front-month/roll limitation được document và roll được xử lý bằng calendar proxy trong economic evaluation.

2. **Stock market simulator — COMPLETE**  
   Có portfolio notional, integer contract sizing, 1:2 project margin, HNX/VSDC, broker fee, PIT tax, slippage, daily MTM và roll proxy.

3. **Calibrate Heston model trên dữ liệu VN30 — COMPLETE WITH SCOPE QUALIFIER**  
   Hoàn thành physical-measure historical Heston calibration/estimation và validation/freeze pipeline; không phải option-surface risk-neutral calibration.

Các qualifier/limitations ở trên là một phần của methodology chính thức, không phải phần phụ có thể bỏ khỏi báo cáo.
