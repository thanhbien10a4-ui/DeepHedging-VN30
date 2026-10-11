# Kiểm chứng trên Antigravity

## Prompt giao việc
> Kiểm tra DeepHedging-VN30 trên máy này sau git pull. Ghi git status, source commit và interpreter. Không ghi đè checkpoints/datasets; không train, tune hoặc đổi features/objective/clipping/simulator theo TEST. Chạy python scripts/project.py audit; đọc missing/errors và tìm artifacts cũ trong thư mục chạy hoặc backup. Chạy python scripts/project.py report. Nếu cần frozen evaluation, sao lưu output cũ rồi chạy python scripts/project.py evaluate và ghi commands/environment/hashes. Đọc training curves, sanity, best_epoch, baseline/OOS/cost/regime/stress. Không gọi artifact checks là toàn bộ thí nghiệm PASS. Khi đủ bằng chứng, chạy python scripts/project.py package --include-models. Nếu thiếu, chỉ dùng --allow-incomplete tạo diagnostic bundle và nói rõ chưa đầy đủ. Gửi ZIP, source commit, báo cáo và danh sách còn thiếu. Không tự tick checklist để che khoảng trống.

## Tiêu chí nghiệm thu
- Raw/aligned/splits khớp số dòng/ngày; state không dùng tương lai; scaler fit TRAIN.
- Synthetic metadata/NPZ shapes đúng; params từ TRAIN/VALIDATION frozen config.
- Training history hữu hạn, trọng số có cập nhật, checkpoint theo VALIDATION; early stopping không phải proof of global convergence.
- Fine-tune xác nhận best_epoch; epoch 0 không được gọi improvement.
- Baselines/proposed model cùng exposure/costs/constraints; giữ failure modes và episode-dependence caveat.
- Simulator nhất quán config; không diễn giải margin 50% là historical schedule.
- ZIP có manifest hashes, không secrets; nguồn checkpoint đáng tin cậy.

Test công cụ bàn giao: `python -m unittest discover -s tests -v`. Test này không thay thế kiểm chứng ML. Sau nghiệm thu, chỉ cập nhật các ô checklist có đủ bằng chứng; còn thiếu actual liquidity analysis, multiple seeds và confidence intervals phải nêu rõ.
