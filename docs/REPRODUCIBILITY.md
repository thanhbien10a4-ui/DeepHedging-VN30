# Tái lập và quản lý bằng chứng

Máy có kết quả: audit → report → package. Không train lại trước khi tìm artifacts cũ. Tái lập độc lập phải ở clone riêng, ghi source commit/environment, không thay thế âm thầm kết quả đã mở TEST.

Chạy từng script bằng `python <path>` từ repo root, dừng khi lỗi. Thứ tự final:

| Bước | Script |
|---|---|
| 01 | src/preprocessing/01_validate_and_align.py |
| 02 | src/preprocessing/02_feature_engineering.py |
| 03 | src/preprocessing/03_statistical_validation.py |
| 04 | src/preprocessing/04_rollover_outlier_diagnostics.py |
| 09 | src/preprocessing/09_prepare_chronological_splits.py |
| 10 | src/simulator/10_calibrate_train_only.py |
| 11 | src/simulator/11_select_simulator_on_validation.py |
| 12 | src/simulator/12_tune_simulator_on_validation.py |
| 13 | src/simulator/13_finalize_validation_refinement.py |
| 14 | src/simulator/14_generate_frozen_synthetic.py |
| 15 | src/hedging/15_baseline_hedging_validation.py |
| 16 | src/models/16_prepare_deep_hedging_sequences.py |
| 17 | src/models/17_train_lstm_variance_baseline.py |
| 18 | src/models/18_train_lstm_cvar_turnover.py |
| 19 | src/models/19_prepare_real_deep_hedging_sequences.py |
| 20 | src/models/20_diagnose_real_domain_shift.py |
| 21 | src/models/21_zero_shot_real_transfer_diagnostic.py |
| 22 | src/models/22_finetune_lstm_on_real_train.py |
| 23 | src/evaluation/23_final_oos_real_test.py |
| 24 | src/evaluation/24_cost_aware_oos_frozen.py |
| 25 | src/evaluation/25_market_regime_analysis_frozen.py |
| 26 | src/evaluation/26_transaction_cost_ablation_frozen.py |
| 27 | src/evaluation/27_graceful_degradation_frozen.py |
| 28 | src/evaluation/28_cumulative_performance_reporting.py |
| 29 | src/utils/29_capture_environment.py |
| 30 | src/evaluation/30_full_market_simulator_frozen.py |

05/05b/06/07/08 là exploratory legacy full-sample diagnostics, không cần cho pipeline final 10–14. Không dùng để lựa chọn lại theo TEST. Xem contamination caveat trong methodology.

## Bằng chứng
Lưu commit, commands, timestamps, seed, dataset/checkpoint hashes, training histories, sanity reports và metrics/figures. SHA-256 manifest chứng minh bytes bundle, không tự chứng minh file được tạo từ commit nào; cần run log để xác nhận provenance.

Audit kiểm tra sự hiện diện/định dạng/sanity, không chứng minh convergence hay superiority. Đọc curves, best_epoch và failure modes trước nghiệm thu. Checkpoint phải từ nguồn tin cậy vì pipeline dùng torch.load(weights_only=False). Generated data/model thường gitignored: tìm output/backups trước khi tạo mới.
