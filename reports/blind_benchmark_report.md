# BÁO CÁO BENCHMARK ĐỘC LẬP (BLIND TEST) - TOXICITY 4 MỨC
## MÔ HÌNH: fastText 300d + Bi-GRU-LSTM-CNN (ĐA NHIỆM)

> [!IMPORTANT]
> - Phân loại Toxicity ở chế độ 4 mức chi tiết (0: Không độc, 1: Thiếu lịch sự, 2: Xúc phạm, 3: Thù ghét/Đe dọa).

## 1. Bảng Xếp Hạng Hiệu Năng Đa Seed (Model Leaderboard)

| Run / Seed | Sentiment Macro F1 | Toxicity (4 Lớp) Macro F1 | **Combined Macro F1** | Thông lượng (mẫu/s) | Độ trễ (ms/câu) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `bigru_fasttext_seed1` | 0.6315 | 0.6216 | **0.6266** | 776 | 1.29 ms |
| `bigru_fasttext_seed2` | 0.6456 | 0.4993 | **0.5725** | 1463 | 0.68 ms |
| `bigru_fasttext_seed3` | 0.6498 | 0.5984 | **0.6241** | 1458 | 0.69 ms |
| **TRUNG BÌNH (Mean ± Std)** | **0.6423 ± 0.0079** | **0.5731 ± 0.0531** | **0.6077 ± 0.0250** | **1232 mẫu/s** | **0.89 ms** |

---

## 2. Phân Tích Chi Tiết Trên Run Tiêu Biểu (bigru_fasttext_seed1)

### 2.1. Trục 1: Phân Loại Cảm Xúc (Sentiment - 3 Lớp)
- **Macro F1:** **`0.6315`** | **Sai số ECE:** `0.2440`
```text
               precision    recall  f1-score      support
Negative (-1)   0.700138  0.474371  0.565556  1073.000000
Neutral (0)     0.578701  0.747405  0.652322  1156.000000
Positive (1)    0.673550  0.679563  0.676543  1008.000000
accuracy        0.635774  0.635774  0.635774     0.635774
macro avg       0.650796  0.633780  0.631473  3237.000000
weighted avg    0.648490  0.635774  0.631103  3237.000000
```

### 2.2. Trục 2: Phân Loại Độ Độc Hại (Toxicity - 4 Lớp)
- **Macro F1:** **`0.6216`** | **Sai số ECE:** `0.0776`
```text
                   precision    recall  f1-score      support
T0_NonToxic (0)     0.939099  0.952485  0.945745  2736.000000
T1_Unpolite (1)     0.549828  0.509554  0.528926   314.000000
T2_Insult (2)       0.646154  0.531646  0.583333   158.000000
T3_HateThreat (3)   0.365854  0.517241  0.428571    29.000000
accuracy            0.885079  0.885079  0.885079     0.885079
macro avg           0.625234  0.627732  0.621644  3237.000000
weighted avg        0.881904  0.885079  0.882989  3237.000000
```

### 2.3. Tỷ Lệ Báo Nhầm Toxic (Đoán T1-T3 trên câu T0) Theo Nhóm Cảm Xúc
| Nhóm Cảm Xúc | Tỷ lệ báo nhầm (FPR) |
| :--- | :---: |
| **NEGATIVE** | **8.60%** |
| **NEUTRAL** | **4.43%** |
| **POSITIVE** | **2.51%** |
