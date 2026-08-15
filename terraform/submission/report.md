# LAB 16 - AWS Cloud AI Environment

Lab được triển khai trên AWS tại region us-east-1 với instance t3.micro.

Dataset Credit Card Fraud Detection gồm 284807 dòng dữ liệu.
Thời gian load dữ liệu là 2.5771 giây và thời gian training là 1.8792 giây.
Mô hình đạt AUC-ROC 0.902379 và Accuracy 0.977301.
F1-score đạt 0.1162, Precision 0.062271 và Recall 0.867347.
Inference latency là 1.1729 ms, throughput khoảng 302193.68 dòng/giây.
Resource usage được kiểm tra bằng top, free -h và ip -s link.
Sau khi hoàn tất, tài nguyên AWS được dọn dẹp bằng terraform destroy.
