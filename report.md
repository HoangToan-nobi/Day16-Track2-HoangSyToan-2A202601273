# Báo cáo ngắn — Day 16 GCP CPU

- Dataset Credit Card Fraud Detection gồm 284.807 dòng và 30 đặc trưng.
- Thời gian đọc dữ liệu là 1,9433 giây.
- Thời gian huấn luyện LightGBM là 1,7188 giây, với best iteration bằng 1.
- Mô hình đạt AUC-ROC 0,902379 và Accuracy 0,977301.
- Recall đạt 0,867347, cho thấy mô hình phát hiện được phần lớn giao dịch gian lận.
- Precision 0,062271 và F1-Score 0,1162 còn thấp do dữ liệu mất cân bằng mạnh.
- Inference một dòng có latency khoảng 0,887 ms.
- Throughput inference đạt khoảng 901.017 dòng/giây trên VM CPU e2-medium.
