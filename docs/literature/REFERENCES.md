# Tài liệu tham khảo cho đồ án Deep Hedging VN30

Danh mục này ưu tiên các nguồn nền tảng về Deep Hedging, stochastic volatility, CVaR, hedge-ratio estimation, volatility estimation và các nguồn chính thức/đặc thù của thị trường phái sinh Việt Nam. Danh mục hiện có **22 nguồn**, vượt yêu cầu tối thiểu 15 tài liệu.

## A. Deep Hedging, deep learning và risk measures

1. **Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019).** Deep hedging. *Quantitative Finance, 19*(8), 1271–1291. https://doi.org/10.1080/14697688.2019.1571683  
   **Vai trò trong đồ án:** tài liệu nền tảng trực tiếp cho framework Deep Hedging, market frictions và convex risk measures.

2. **Carbonneau, A., & Godin, F. (2021).** Equal risk pricing of derivatives with deep hedging. *Quantitative Finance, 21*(4), 593–608. https://doi.org/10.1080/14697688.2020.1806343  
   **Vai trò:** mở rộng ứng dụng Deep Hedging với convex risk measures và nhiều dynamics khác nhau.

3. **Hochreiter, S., & Schmidhuber, J. (1997).** Long short-term memory. *Neural Computation, 9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735  
   **Vai trò:** nền tảng kiến trúc LSTM dùng cho policy network.

4. **Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999).** Coherent measures of risk. *Mathematical Finance, 9*(3), 203–228. https://doi.org/10.1111/1467-9965.00068  
   **Vai trò:** nền tảng lý thuyết cho coherent/convex risk measures.

5. **Rockafellar, R. T., & Uryasev, S. (2002).** Conditional value-at-risk for general loss distributions. *Journal of Banking & Finance, 26*(7), 1443–1471. https://doi.org/10.1016/S0378-4266(02)00271-6  
   **Vai trò:** cơ sở lý thuyết và tối ưu cho CVaR, objective chính của mô hình final.

## B. Heston, volatility và time-series dynamics

6. **Heston, S. L. (1993).** A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies, 6*(2), 327–343. https://doi.org/10.1093/rfs/6.2.327  
   **Vai trò:** mô hình stochastic volatility dùng làm structural synthetic-market baseline.

7. **Engle, R. F. (1982).** Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica, 50*(4), 987–1007. https://doi.org/10.2307/1912773  
   **Vai trò:** nền tảng ARCH, giải thích volatility clustering.

8. **Bollerslev, T. (1986).** Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics, 31*(3), 307–327. https://doi.org/10.1016/0304-4076(86)90063-1  
   **Vai trò:** GARCH và conditional volatility dynamics.

9. **Garman, M. B., & Klass, M. J. (1980).** On the estimation of security price volatilities from historical data. *The Journal of Business, 53*(1), 67–78. https://doi.org/10.1086/296072  
   **Vai trò:** Garman–Klass variance estimator dùng trong preprocessing/Heston proxy.

10. **Parkinson, M. (1980).** The extreme value method for estimating the variance of the rate of return. *The Journal of Business, 53*(1), 61–65. https://doi.org/10.1086/296071  
    **Vai trò:** range-based volatility estimation, tài liệu so sánh cho GK estimator.

## C. Futures hedging và minimum-variance hedge ratio

11. **Ederington, L. H. (1979).** The hedging performance of the new futures markets. *The Journal of Finance, 34*(1), 157–170. https://doi.org/10.1111/j.1540-6261.1979.tb02077.x  
    **Vai trò:** tài liệu kinh điển về hedging effectiveness và futures hedge ratio.

12. **Baillie, R. T., & Myers, R. J. (1991).** Bivariate GARCH estimation of the optimal commodity futures hedge. *Journal of Applied Econometrics, 6*(2), 109–124. https://doi.org/10.1002/jae.3950060202  
    **Vai trò:** dynamic optimal hedge ratio dựa trên conditional covariance/variance.

13. **Kroner, K. F., & Sultan, J. (1993).** Time-varying distributions and dynamic hedging with foreign currency futures. *Journal of Financial and Quantitative Analysis, 28*(4), 535–551. https://doi.org/10.2307/2331164  
    **Vai trò:** dynamic hedging, time-varying distributions và transaction costs.

14. **Figlewski, S. (1985).** Hedging with stock index futures: Theory and application in a new market. *Journal of Futures Markets, 5*(2), 183–199. https://doi.org/10.1002/fut.3990050204  
    **Vai trò:** trực tiếp liên quan đến stock-index futures hedging.

## D. Nghiên cứu liên quan thị trường VN30/Vietnam derivatives

15. **Truong, L. D., & Friday, H. S. (2021).** The impact of the introduction of index futures on the daily returns anomaly in the Ho Chi Minh Stock Exchange. *International Journal of Financial Studies, 9*(3), 43. https://doi.org/10.3390/ijfs9030043  
    **Vai trò:** bằng chứng học thuật về ảnh hưởng của VN30 index futures lên thị trường cơ sở Việt Nam.

16. **Nguyen, A. T. K., Truong, L. D., & Friday, H. S. (2022).** Expiration-day effects of index futures in a frontier market: The case of Ho Chi Minh Stock Exchange. *International Journal of Financial Studies, 10*(1), 3. https://doi.org/10.3390/ijfs10010003  
    **Vai trò:** hỗ trợ thảo luận expiry effects và lý do cần chú ý roll/expiry mechanics.

## E. Nguồn chính thức và pháp lý cho VN30 futures

17. **Sở Giao dịch Chứng khoán Hà Nội (HNX).** *Sản phẩm giao dịch – Mẫu Hợp đồng Tương lai Chỉ số VN30.*  
    https://cms.hnx.vn/vi-vn/huong-dan/chi-tiet-thu-tuc-36-65.html  
    **Vai trò:** nguồn chính thức cho multiplier 100.000 VND/điểm, tick size 0,1, biên độ ±7%, tháng đáo hạn và ngày giao dịch cuối cùng.

18. **Sở Giao dịch Chứng khoán Hà Nội (HNX) (2017).** *On August 10th, the Vietnam derivatives market will be officially launched.*  
    https://hnx.vn/en-gb/chi-tiet-su-kien-60000956-0-hnx.html  
    **Vai trò:** mốc ra mắt thị trường phái sinh Việt Nam và VN30 futures.

19. **Bộ Tài chính (2017).** *Công văn 11133/BTC-CST về chính sách thuế đối với thu nhập từ chuyển nhượng hợp đồng tương lai trên thị trường chứng khoán phái sinh.*  
    https://thuvienphapluat.vn/cong-van/Thue-Phi-Le-Phi/Cong-van-11133-BTC-CST-2017-chinh-sach-thue-thu-nhap-tu-chuyen-nhuong-hop-dong-tuong-lai-366517.aspx  
    **Vai trò:** công thức thuế TNCN 0,1% trên giá chuyển nhượng từng lần, trong đó có hệ số ký quỹ ban đầu.

20. **Bộ Tài chính (2025).** *Quyết định 1541/QĐ-BTC ngày 29/04/2025 về mức giá dịch vụ trong lĩnh vực chứng khoán.*  
    Bản công bố tại Ủy ban Chứng khoán Nhà nước: https://ssc.gov.vn/webcenter/portal/ubck/pages_r/l/chitit?dDocName=APPSSCGOVVN1620154432  
    **Vai trò:** mức phí dịch vụ chính thức tại VNX/HNX/VSDC, gồm phí giao dịch phái sinh và phí bù trừ.

21. **Sở Giao dịch Chứng khoán Hà Nội (HNX) (2025).** *Thông báo về việc chính thức đưa vào vận hành Hệ thống công nghệ thông tin thị trường chứng khoán Việt Nam.*  
    https://hnx.vn/vi-vn/m-tin-tuc-hnx/Thong%20bao%20ve%20viec%20chinh%20thuc%20dua%20vao%20van%20hanh%20He%20thong%20cong%20nghe%20thong%20tin%20thi%20truong%20chung%20khoan%20Viet%20Nam-60020925-1.html  
    **Vai trò:** xác nhận KRX go-live từ 05/05/2025.

22. **Vietnam Securities Depository and Clearing Corporation (VSDC) (2024).** *Initial margin rate for VN30 Index Futures effective from 16/02/2024.*  
    https://vsdc.vn/en/ad/167471  
    **Vai trò:** ví dụ chính thức về tỷ lệ initial margin 17% trong năm 2024; không được dùng để suy diễn toàn bộ lịch sử 2017–2026.

## Ghi chú sử dụng trong báo cáo

- Khi mô tả **Deep Hedging / market frictions / convex risk measure**, ưu tiên [1], [4], [5].
- Khi mô tả **LSTM policy**, dùng [3].
- Khi mô tả **Heston/stochastic volatility**, dùng [6], đồng thời ghi rõ đồ án chỉ có historical underlying data nên đang làm **physical-measure estimation**, không phải classical risk-neutral option-surface calibration.
- Khi giải thích **volatility clustering**, dùng [7], [8].
- Khi biện minh cho **Garman–Klass/range-based proxy**, dùng [9], [10].
- Khi mô tả **minimum-variance/dynamic hedge ratio**, dùng [11]–[14].
- Khi bàn về **VN30 futures, expiry/roll**, dùng [15]–[18], [21].
- Khi bàn về **fees/tax/margin**, dùng [19], [20], [22].

## Cảnh báo phương pháp

Danh mục tài liệu không làm thay đổi quy tắc khóa TEST: real TEST 2025–2026 đã mở sau khi model được freeze, nên không được dùng các tài liệu đọc bổ sung sau thời điểm này để retune model/hyperparameter theo kết quả TEST. Các nguồn bổ sung chỉ dùng để giải thích phương pháp, giới hạn, market mechanics và viết báo cáo.
