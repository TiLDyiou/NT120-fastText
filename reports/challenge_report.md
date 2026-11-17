# BÁO CÁO ĐÁNH GIÁ TẬP MÙ CHALLENGE (PHASE 7) — $N = 3,237$ CÂU

> **Lưu ý:** Chạy **đúng 1 lần duy nhất** trên tập mù `challenge.csv` sau khi đóng băng toàn bộ mô hình và hậu xử lý ở Phase 6. Tuyệt đối không chỉnh sửa và chấm lại.

## 1. Bảng Điểm Tổng hợp 3 Seed (Mean ± Std)

| Seed | Sentiment Macro-F1 | **Toxicity Macro-F1 (4 lớp)** | Combined Macro-F1 | QWK (Toxicity) | ECE Sent | ECE Tox | CI 95% Tox F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `seed_1` | 0.6431 | **0.6635** | 0.6533 | 0.6731 | 0.0782 | 0.0203 | [0.6159, 0.7092] |
| `seed_2` | 0.6376 | **0.6568** | 0.6472 | 0.6769 | 0.0816 | 0.0174 | [0.6091, 0.7017] |
| `seed_3` | 0.6510 | **0.6575** | 0.6543 | 0.6765 | 0.0676 | 0.0140 | [0.6141, 0.7022] |
| **Mean ± Std** | **0.6439 ± 0.0067** | **0.6593 ± 0.0037** | **0.6516 ± 0.0038** | **0.6755 ± 0.0021** | **0.0758 ± 0.0073** | **0.0172 ± 0.0031** | — |

## 2. Chi tiết từng Lớp (Mean ± Std qua 3 Seed)

### 2.1. Nhiệm vụ Sentiment (3 lớp: `-1, 0, 1`)
| Lớp | F1-Score (Mean ± Std) | Support |
| :--- | :---: | :---: |
| `Negative (-1)` | **0.5241 ± 0.0188** | 1,073 |
| `Neutral (0)` | **0.6917 ± 0.0006** | 1,156 |
| `Positive (1)` | **0.7158 ± 0.0015** | 1,008 |

### 2.2. Nhiệm vụ Toxicity (4 lớp: `0, 1, 2, 3`)
| Lớp | F1-Score (Mean ± Std) | Support |
| :--- | :---: | :---: |
| `T0_NonToxic` | **0.9522 ± 0.0013** | 2,736 |
| `T1_Unpolite` | **0.5665 ± 0.0076** | 314 |
| `T2_Insult` | **0.6298 ± 0.0109** | 158 |
| `T3_HateThreat` | **0.4886 ± 0.0129** | 29 |

> **Nguyên tắc diễn giải (§7.2.2):** Do lớp 3 (`T3_HateThreat`, support = 29) rất hiếm, F1 riêng lớp 3 dao động mạnh chỉ với vài mẫu đoán sai. Mọi kết luận đánh giá mô hình chỉ rút ra từ **`Macro-F1` tổng 4 lớp** và **`Confusion matrix` $4 \times 4$**.

### 2.3. Ma trận Nhầm lẫn (Seed 1 đại diện)
- **Sentiment `3×3` (hàng = thực tế `[-1, 0, 1]`, cột = dự đoán):**
```text
[[412 477 184]
 [ 47 980 129]
 [ 52 218 738]]
```
- **Toxicity `4×4` (hàng = thực tế `[0, 1, 2, 3]`, cột = dự đoán):**
```text
[[2662   55   13    6]
 [ 137  157   14    6]
 [  44   23   89    2]
 [   9    2    4   14]]
```

## 3. Tỷ lệ Báo nhầm Mức độ Độc hại (FPR) trên tập Non-toxic (`T0`) theo Nhóm Cảm xúc

> **Lưu ý:** Đây là định nghĩa FPR đa mức riêng biệt ($0 \to 1$, $0 \to 2$, $0 \to 3$), khác bản chất với định nghĩa nhị phân cũ nên không so sánh trực tiếp số học với báo cáo nhị phân trước đây.

| Nhóm Cảm xúc | Support `T0` | $\text{FPR}_{0 \to 1}$ (Mean ± Std) | $\text{FPR}_{0 \to 2}$ (Mean ± Std) | $\text{FPR}_{0 \to 3}$ (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: |
| `Negative (-1)` | 651 | 4.71% ± 0.90% | 1.43% ± 0.23% | 0.67% ± 0.18% |
| `Neutral (0)` | 1,128 | 1.80% ± 0.45% | 0.56% ± 0.14% | 0.21% ± 0.10% |
| `Positive (1)` | 957 | 1.57% ± 0.31% | 0.07% ± 0.06% | 0.03% ± 0.06% |

## 4. Thông lượng & Độ trễ Suy luận (Cold Start vs. Warm Start)

- **Phần cứng:** GPU `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (VRAM `6.0 GB`), CPU `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel`, RAM `19.71 GB`, `batch_size = 256`, số token OOV mới trong `challenge` = `1079`.

| Kịch bản | Thời gian TB (s) | Thông lượng (câu/s) | Độ trễ (ms/câu) |
| :--- | :---: | :---: | :---: |
| **Cold start** (lần đầu tra `.bin` cho token mới) | 1.325 | **2621 ± 774** | **0.409 ± 0.142** |
| **Warm start** (đã có cache bộ nhớ) | 0.340 | **9537 ± 565** | **0.105 ± 0.006** |

## 5. Đối chiếu với Bản Cũ (Keras `Bi-GRU-LSTM-CNN` 4 lớp trong `NT120-fastText/reports/blind_benchmark_report.md`)

| Chỉ số trên `challenge` (3,237 câu) | Bản cũ v1 (Keras + `.vec` + `pyvi`) | **Bản mới v2 (PyTorch + `.bin` + Song song)** |
| :--- | :---: | :---: |
| **Sentiment Macro-F1 (3 lớp)** | 0.6423 ± 0.0079 | **0.6439 ± 0.0067** |
| **Toxicity Macro-F1 (4 lớp)** | 0.5731 ± 0.0531 | **0.6593 ± 0.0037** |
| **Combined Macro-F1** | 0.6077 ± 0.0250 | **0.6516 ± 0.0038** |
| **ECE Sentiment (Seed 1)** | 0.2440 | **0.0782** |
| **ECE Toxicity (Seed 1)** | 0.0776 | **0.0203** |
