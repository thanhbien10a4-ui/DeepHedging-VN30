# Ứng dụng thực tiễn: hỗ trợ phòng hộ danh mục VN30

## Người dùng và phạm vi
Nhà đầu tư/tổ chức đã có long exposure gần VN30 muốn giảm downside risk trong một horizon 60 phiên. Mô hình xuất hedge ratio tham khảo; người quản lý quyết định giao dịch. Chưa cung cấp executable trading recommendation cho danh mục cá nhân, vì thiếu beta/tracking-error estimation, tài khoản margin và dữ liệu hợp đồng cụ thể.

## Quy trình triển khai thử
1. Xác định giá trị và mức tương đồng của danh mục với VN30. Danh mục mid-cap/penny không mặc nhiên hedge tốt bằng VN30F1M.
2. Chuẩn bị dữ liệu point-in-time đã biết ở thời điểm quyết định; kiểm tra missing/outlier, basis và đúng contract.
3. Nạp đúng checkpoint/scaler đã freeze. Không fit lại scaler bằng ngày TEST hay đổi state schema.
4. Chạy inference h_t; quy đổi N=round-half-up(h_t V_t/(F_t M)). Long spot tương ứng short futures. Kiểm tra realized hedge ratio sau rounding.
5. Kiểm tra cash, initial margin, variation margin buffer và position limits ngoài mạng. Nếu không đủ vốn, giảm exposure hoặc từ chối đề xuất; không coi giới hạn h≤1.5 là đủ kiểm soát margin.
6. Lập lịch rebalance và roll trên contract thực. Đo phí, spread, slippage và funding; giả định 1 tick/50% margin trong nghiên cứu không tự áp dụng vào live.
7. Paper trade và ghi state/action/filled quantity/PnL/cost, so với hedge 1:1. Xác định trước reset LSTM/episode clock; bản hiện tại chưa có continuous deployment policy.
8. Kiểm tra domain shift và downside scenarios. Không tự retrain hoặc thay model vì vài ngày thua; dùng protocol lựa chọn trên dữ liệu mới tách riêng.

## Ví dụ minh họa, không phải tín hiệu giao dịch
V=10 tỷ, F=2.000, M=100.000, h=0.8 → short 40 hợp đồng. Theo giả định 50%, initial margin khoảng 4 tỷ VND, ngoài exposure spot. Phí và variation margin còn cần vốn bổ sung. Điều này minh họa vì sao 10 tỷ exposure không có nghĩa chỉ cần tổng vốn 10 tỷ để vận hành chiến lược.

## Điều kiện trước giao dịch thật
Broker/exchange specifications được xác minh theo ngày; portfolio beta và cổ tức; execution timing; actual roll spread; hard cash constraints; monitoring và fallback; kiểm chứng độc lập. Không suy ra xác suất thành công hoặc cam kết lợi nhuận từ 370 episode chồng lấn.

## Sản phẩm hiện có
Simulator, checkpoint interface, công cụ audit/report/package là sản phẩm nghiên cứu. Live order routing, broker connection, dashboard giao dịch và quyết định cấp vốn chưa triển khai. Ứng dụng được trình bày đầy đủ về quy trình và giới hạn; chưa tuyên bố triển khai live.
