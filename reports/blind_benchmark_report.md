# BÁO CÁO BENCHMARK ĐỘC LẬP (BLIND TEST) TRÊN TẬP CHALLENGE
## MÔ HÌNH: fastText 300d + Bi-GRU-LSTM-CNN (ĐA NHIỆM)

> [!IMPORTANT]
> **Tuân thủ nguyên tắc Blind Test tuyệt đối:**
> - Mô hình hoàn toàn **chỉ gọi suy luận `predict()`**, tuyệt đối không fit lại.
> - Tập Challenge độc lập hoàn toàn với quá trình huấn luyện ($0$ mẫu rò rỉ sau khi qua bộ lọc Mục 10 Guideline).

## 1. Thông Tin Tập Dữ Liệu Kiểm Thử Mù
- **Đường dẫn tập Challenge:** `merged/challenge/challenge.csv`
- **Tổng số mẫu:** **3,237 bình luận**
- **Số lượng Seed kiểm định:** **3 runs** (bigru_fasttext_seed1, bigru_fasttext_seed2, bigru_fasttext_seed3)

## 2. Bảng Xếp Hạng Hiệu Năng Đa Seed (Model Leaderboard)

| Run / Seed | Sentiment Macro F1 | Toxicity Macro F1 | **Combined Macro F1** | Thông lượng (mẫu/s) | Độ trễ (ms/câu) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `bigru_fasttext_seed1` | 0.6644 | 0.8214 | **0.7429** | 771 | 1.30 ms |
| `bigru_fasttext_seed2` | 0.6565 | 0.8292 | **0.7428** | 973 | 1.03 ms |
| `bigru_fasttext_seed3` | 0.6607 | 0.8325 | **0.7466** | 1308 | 0.76 ms |
| **TRUNG BÌNH (Mean ± Std)** | **0.6605 ± 0.0032** | **0.8277 ± 0.0047** | **0.7441 ± 0.0018** | **1017 mẫu/s** | **1.03 ms** |

---

## 3. Phân Tích Chi Tiết Trên Run Tiêu Biểu (bigru_fasttext_seed1)

### 3.1. Trục 1: Phân Loại Cảm Xúc (Sentiment - 3 Lớp)
- **Macro F1:** **`0.6644`** | **Sai số hiệu chuẩn ECE:** `0.0838`
```text
               precision    recall  f1-score      support
Negative (-1)   0.722222  0.533085  0.613405  1073.000000
Neutral (0)     0.621545  0.758651  0.683288  1156.000000
Positive (1)    0.687621  0.705357  0.696376  1008.000000
accuracy        0.667285  0.667285  0.667285     0.667285
macro avg       0.677129  0.665697  0.664356  3237.000000
weighted avg    0.675493  0.667285  0.664199  3237.000000
```

### 3.2. Trục 2: Phân Loại Độ Độc Hại (Toxicity - 2 Lớp)
- **Macro F1:** **`0.8214`** | **Sai số hiệu chuẩn ECE:** `0.0055`
```text
               precision    recall  f1-score      support
Non-toxic (0)   0.945421  0.943348  0.944383  2736.000000
Toxic (1)       0.694280  0.702595  0.698413   501.000000
accuracy        0.906086  0.906086  0.906086     0.906086
macro avg       0.819851  0.822971  0.821398  3237.000000
weighted avg    0.906551  0.906086  0.906314  3237.000000
```

### 3.3. Tỷ Lệ Báo Nhầm Toxic Theo Nhóm Cảm Xúc (FPR by Emotion)
| Nhóm Cảm Xúc | Tỷ lệ báo nhầm (FPR) |
| :--- | :---: |
| **NEGATIVE** | **10.91%** |
| **NEUTRAL** | **5.23%** |
| **POSITIVE** | **2.61%** |

---

## 5. Kết Luận So Sánh Với TF-IDF
1. **Độ ổn định đa Seed:** Neural Network Bi-GRU-LSTM-CNN đạt điểm F1 trung bình là `0.7441 ± 0.0018`, chứng minh tính hội tụ bền vững.
2. **Tốc độ suy luận:** Đạt `1017 mẫu/giây` (độ trễ `1.03 ms/câu`), hoàn toàn đáp ứng yêu cầu lọc nội dung thời gian thực.
3. **File dự đoán:** Đã xuất chi tiết tại `reports/preds_challenge_*.csv` để phục vụ phân tích lỗi định tính.
