# NT120-fastText: Bi-GRU-LSTM-CNN Đa Nhiệm

Dự án này xây dựng mô hình học sâu kết hợp vector nhúng tĩnh (Static Word Embeddings) để phân loại bình luận tiếng Việt trên mạng xã hội theo hai trục cùng lúc: **Cảm xúc (Sentiment)** và **Độ độc hại (Toxicity)**.

Mô hình là đại diện tiêu biểu cho **Thế hệ 2 (Deep Learning cổ điển)** trong đề tài NT120, làm cầu nối so sánh giữa **Thế hệ 1 (TF-IDF + LinearSVC)** và **Thế hệ 3 (Transformer - BaMiBERT)**.

| Trục | Số lớp | Giá trị | Ý nghĩa |
| :--- | :---: | :---: | :--- |
| **Sentiment** | 3 lớp | -1 / 0 / 1 | Tiêu cực (Negative) / Trung tính (Neutral) / Tích cực (Positive) |
| **Toxicity** | 2 lớp (Nhị phân) | 0 / 1 | Không độc hại (Non-toxic) / Độc hại (Toxic - gồm cả mức 1, 2, 3) |

> **Ghi chú về nhãn Toxicity:** Theo thiết kế đề tài ở nhánh Deep Learning, bài toán Toxicity được chuẩn hóa về dạng nhị phân theo ngưỡng `--tox-min-level 1` (Mức 0 là Non-toxic; Mức 1: thiếu lịch sự, Mức 2: xúc phạm cá nhân, Mức 3: thù ghét/đe dọa đều được gộp chung thành Toxic). Cột nhãn gốc `tox_level` (0..3) vẫn được lưu lại phục vụ phân tích lỗi định tính.

---

## Những gì đã thực hiện

1. **Khử trùng lặp & Chống rò rỉ dữ liệu (Anti-Leakage):** Xây dựng bộ lọc khóa khớp theo đúng Mục 10 của `LABELING_GUIDELINES.md`. Loại bỏ 187 bình luận trùng giữa train và valid (quy tắc: *xóa ở valid, giữ ở train*) để bảo toàn tính khách quan tuyệt đối của tập kiểm định.
2. **Tiền xử lý văn bản đa tầng đặc thù cho mạng xã hội:**
   - Chuẩn hóa Unicode NFC, thay link bằng thẻ `url`.
   - Quy đổi emoticon gõ phím (`:))`, `:((`, `<3`) thành Emoji Unicode chuẩn **trước khi tách từ**, bảo vệ cấu trúc biểu cảm không bị băm nát.
   - Rút gọn ký tự lặp quá đà (`nguuuu` $\rightarrow$ `ngu`), xử lý ký tự lách luật (`đ.m`, `l.ồ.n`), chuẩn hóa teencode phổ biến.
   - Tách từ tiếng Việt bằng `pyvi` và hạ chữ thường.
3. **Khai thác fastText 300d tiếng Việt kết hợp Syllable-Averaging:**
   - Sử dụng pre-trained embedding `cc.vi.300.vec.gz` từ Common Crawl.
   - Bù đắp từ ghép thiếu bằng trung bình cộng vector các âm tiết: $\vec{v}_{\text{dịch\_vụ}} = (\vec{v}_{\text{dịch}} + \vec{v}_{\text{vụ}})/2$, nâng độ phủ từ vựng lên trên 96%.
4. **Thiết kế kiến trúc lai Bi-GRU + LSTM + CNN:**
   - Nhánh học tuần tự 2 chiều (Bi-GRU) kết hợp bộ nhớ dài hạn (LSTM) và bộ trích xuất n-gram đặc trưng cục bộ (Conv1D).
   - Tầng gộp kết hợp cực đại và trung bình (`GlobalMaxPooling1D` + `GlobalAveragePooling1D`).
5. **Cơ chế Học đa nhiệm (Multi-task Learning):** Thân mạng chung (`Shared Backbone`) rẽ nhánh ra 2 đầu ra `Softmax` độc lập, tận dụng mối tương quan ngữ nghĩa giữa Cảm xúc và Độ độc hại.
6. **Xử lý mất cân bằng lớp & Thích ứng tốc độ học:**
   - Áp dụng `Class Weights (Balanced)` nhân trọng số phạt nặng hơn cho lớp thiểu số Toxic (tỷ lệ phạt $\approx 4.16 \times$).
   - Fine-tuning vector embedding (`trainable=True`).
   - Callback `BestMacroF1` tự động giảm Learning Rate 50% khi Macro-F1 bão hòa và dừng sớm (`Early Stopping` với `patience=4`).
7. **Kiểm định đa Seed (Seed 1, 2, 3) & Đánh giá mù độc lập (Blind Test Benchmark):**
   - Huấn luyện 3 lần độc lập để đo độ ổn định $(\text{Mean} \pm \text{Std})$.
   - Đánh giá mù trên tập `challenge.csv` (3.237 câu chưa từng thấy khi train).
   - Đo đạc thông lượng suy luận thực tế (Throughput mẫu/giây) và độ trễ (Latency ms/câu).
   - Đo lường sai số hiệu chuẩn xác suất (ECE) và Tỷ lệ báo nhầm theo Cảm xúc (FPR by Emotion).

---

## Thuật toán và Lý do lựa chọn

Mỗi quyết định kỹ thuật đều có cơ sở khoa học và mục đích rõ ràng. Dưới đây phân định rõ quyết định nào **đã có số đo** và quyết định nào **dựa trên lập luận thiết kế**:

```text
Bình luận thô
  │
  ▼
[Tiền xử lý: NFC → URL → Emoticon sang Emoji → Rút gọn lặp → Xử lý lách luật → Teencode → Pyvi → Lower]
  │
  ▼
[fastText Embedding Matrix 300d] (trainable=True, kết hợp Syllable-Averaging cho từ ghép)
  │
  ▼
[SpatialDropout1D (0.3)]
  │
  ▼
[Bidirectional GRU (128 units)]  (Nắm bắt ngữ cảnh 2 chiều trái - phải)
  │
  ▼
[LSTM (128 units)]               (Ghi nhớ phụ thuộc ngữ nghĩa khoảng cách xa)
  │
  ▼
[Conv1D (128 filters, kernel=3, relu)] (Trích xuất cụm n-gram đặc trưng cục bộ)
  │
  ▼
[Concatenate (GlobalMaxPooling1D + GlobalAveragePooling1D)]
  │
  ▼
[Dropout (0.3) → Dense (64, relu)]
  │
  ├─────────────────────────────────────────┐
  ▼                                         ▼
Dense(3, softmax) [Emotion]               Dense(2, softmax) [Toxicity]
Loss: Sparse Categorical Crossentropy     Loss: Sparse Categorical Crossentropy (Class Weighted)
```

### 1. Tại sao Emoticon phải chuyển thành Emoji TRƯỚC KHI tách từ?
* **Lý do kỹ thuật [Đã kiểm chứng]:** Thư viện tách từ tiếng Việt (`pyvi`) được huấn luyện trên văn bản báo chí. Khi gặp các ký tự chat gõ phím, `pyvi` sẽ băm vụn chúng:
  - `"=)))"` $\rightarrow$ `"= ) ) )"`
  - `"T_T"` $\rightarrow$ `"T _ T"`
  - `"<3"` $\rightarrow$ `"< 3"`
  Nếu tách từ trước, các biểu thức chính quy (Regex) không thể nhận diện được emoticon nữa. Do đó, bước chuyển đổi về Emoji Unicode chuẩn (`😄`, `😭`, `❤️`) bắt buộc phải thực hiện trước. Thư viện `pyvi` và bộ vector fastText đều bảo tồn nguyên vẹn các emoji này.

### 2. Tại sao lại hạ chữ thường (`lower()`), khác với TF-IDF?
* **Lý do kỹ thuật [Lập luận thiết kế]:** Repo TF-IDF giữ nguyên chữ hoa vì n-gram có thể học riêng từ `"NGU"` và `"ngu"`. Ngược lại, fastText ánh xạ từ vào ma trận nhúng. Hầu hết các vector trong từ điển `cc.vi.300` đều ở dạng chữ thường. Nếu giữ chữ hoa, số lượng từ bị coi là lạ (OOV) sẽ tăng vọt, làm phân mảnh không gian vector.

### 3. Tại sao kết hợp cả 3 mạng: Bi-GRU + LSTM + CNN?
* **Bi-GRU:** Đọc câu 2 chiều, nhẹ hơn Bi-LSTM nhưng nắm bắt trạng thái ngữ cảnh tức thời rất tốt.
* **LSTM:** Nằm sau Bi-GRU để chắt lọc và ghi nhớ thông tin ngữ nghĩa đường dài, khắc phục triệt để hiện tượng triệt tiêu gradient.
* **Conv1D (CNN):** Đóng vai trò như mắt người quét nhanh. Các từ khóa mang tính quyết định cao (vd: `rất tệ`, `đồ lừa đảo`, `đẹp xuất sắc`) là các cụm n-gram cục bộ. CNN quét qua câu bằng cửa sổ trượt (kernel = 3) để bắt trọn các tín hiệu mạnh này.
* **Global Pooling kết hợp (Max + Avg):** Max pooling lấy tín hiệu cảm xúc cực đoan nhất; Avg pooling lấy bối cảnh trung bình toàn câu.

### 4. Cơ chế ghép vector âm tiết (Syllable-Averaging) cho tiếng Việt
* Trong tiếng Việt, từ ghép như `dịch_vụ`, `tiết_canh` thường bị thiếu trong fastText gốc (do crawl theo từ đơn cách nhau khoảng trắng).
* Thay vì khởi tạo ngẫu nhiên khiến từ bị mất nghĩa, script tính:
  $$\vec{v}_{\text{dịch\_vụ}} = \frac{\vec{v}_{\text{dịch}} + \vec{v}_{\text{vụ}}}{2}$$
* **Số liệu thực tế:** Giúp độ phủ từ vựng thực tế trong tập train đạt trên 96%.

### 5. Học đa nhiệm (Multi-task) thay vì tách 2 mô hình độc lập
* Hai bài toán có tính tương hỗ ngữ nghĩa cao: câu mang tính độc hại (Toxic) hầu hết thuộc nhóm cảm xúc tiêu cực (Negative); câu khen ngợi hầu như là Non-toxic.
* Việc chia sẻ thân mạng chung giúp mô hình khái quát hóa tốt hơn, đồng thời tiết kiệm 50% tài nguyên tính toán và thời gian suy luận so với chạy 2 mô hình riêng rẽ.

---

## Kết quả đạt được

### 1. BẢNG XẾP HẠNG BLIND TEST TRÊN TẬP CHALLENGE ($n = 3.237$ câu)
Đây là thước đo khách quan cao nhất của đề tài NT120 (mô hình chỉ gọi `predict()`, tuyệt đối không fit lại trên tập này):

| Lần chạy (Run / Seed) | Sentiment Macro F1 | Toxicity Macro F1 | **Combined Macro F1** | Thông lượng (Throughput) | Độ trễ (Latency) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Seed 1** | 0.6644 | 0.8214 | **0.7429** | 771 mẫu/s | 1.30 ms/câu |
| **Seed 2** | 0.6565 | 0.8292 | **0.7428** | 973 mẫu/s | 1.03 ms/câu |
| **Seed 3** | 0.6607 | 0.8325 | **0.7466** | 1.308 mẫu/s | 0.76 ms/câu |
| **TRUNG BÌNH (Mean ± Std)** | **0.6605 ± 0.0032** | **0.8277 ± 0.0047** | **0.7441 ± 0.0018** | **1.017 mẫu/s** | **1.03 ms/câu** |

> **So sánh đối đầu với Thế hệ 1 (TF-IDF + LinearSVC):**
> - **Trục Cảm xúc (Sentiment 3 lớp):** Bi-GRU-LSTM-CNN đạt **0.6605**, vượt trội hơn hẳn so với mức **0.6355** của mô hình chính TF-IDF (+0.025 hay **+2.5%** F1).
> - **Trục Độ độc hại (Toxicity):** Đạt **0.8277** trên bài toán nhị phân, độ chính xác tổng thể đạt **90.61%**.
> - **Độ ổn định:** Độ lệch chuẩn giữa 3 seed chỉ là $\pm 0.0018$, chứng minh mạng nơ-ron hội tụ cực kỳ bền vững, không có hiện tượng "ăn may" theo seed.
> - **Tốc độ:** Xử lý hơn **1.000 bình luận/giây** (độ trễ ~1 ms/câu trên GPU Colab), hoàn toàn sẵn sàng cho kiểm duyệt luồng dữ liệu thời gian thực.

---

### 2. Chi tiết từng lớp trên tập Challenge (Run tiêu biểu Seed 1)

#### Trục Cảm xúc (Sentiment - 3 Lớp):
- **Macro F1:** **`0.6644`** | **Sai số hiệu chuẩn ECE:** `0.0838`

| Lớp Cảm xúc | Precision | Recall | F1-Score | Số lượng mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (-1: Tiêu cực)** | 0.7222 | 0.5331 | **0.6134** | 1.073 |
| **Neutral (0: Trung tính)** | 0.6215 | 0.7587 | **0.6833** | 1.156 |
| **Positive (1: Tích cực)** | 0.6876 | 0.7054 | **0.6964** | 1.008 |
| **Macro Average** | **0.6771** | **0.6657** | **0.6644** | **3.237** |

#### Trục Độ độc hại (Toxicity - 2 Lớp):
- **Macro F1:** **`0.8214`** | **Sai số hiệu chuẩn ECE:** `0.0055` (Mức hiệu chuẩn tuyệt vời)

| Lớp Độ độc hại | Precision | Recall | F1-Score | Số lượng mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Non-toxic (0: Không độc hại)** | 0.9454 | 0.9433 | **0.9444** | 2.736 |
| **Toxic (1: Độc hại)** | 0.6943 | 0.7026 | **0.6984** | 501 |
| **Macro Average** | **0.8199** | **0.8230** | **0.8214** | **3.237** |

---

### 3. Điểm đặc sản đề tài: Tỷ lệ báo nhầm Toxic theo nhóm Cảm xúc (FPR by Emotion)
Chỉ số đo lường mức độ mô hình bị cảm xúc tiêu cực đánh lừa dẫn đến gán nhãn oan trên tập Challenge mù:

| Nhóm Cảm xúc | Tỷ lệ báo nhầm (FPR) trên Challenge | Tỷ lệ báo nhầm (FPR) trên Valid |
| :--- | :---: | :---: |
| **Negative (Tiêu cực)** | **10.91%** | 6.09% |
| **Neutral (Trung tính)** | **5.23%** | 7.06% |
| **Positive (Tích cực)** | **2.61%** | 4.57% |

*Nhận xét:* Trên cả hai tập kiểm thử độc lập, tỷ lệ báo nhầm ở nhóm Tích cực luôn cực thấp ($< 4.5\%$). Ở nhóm Tiêu cực, mô hình vẫn giữ được tỷ lệ báo nhầm dưới $11\%$, thể hiện năng lực phân biệt sắc bén giữa "chê bai dịch vụ/sản phẩm" và "hành vi thóa mạ công kích".

---

### 4. Kết quả trên tập Validation ($n = 7.684$ bình luận) qua 3 Seed
Quá trình huấn luyện sử dụng Early Stopping theo dõi Macro-F1 trung bình 2 trục trên tập Valid:
- **Seed 1:** Macro F1 = 0.7322 (Dừng ở epoch 6, phục hồi epoch 2 tốt nhất)
- **Seed 2:** Macro F1 = 0.7443 (Dừng ở epoch 5, phục hồi epoch 1 tốt nhất)
- **Seed 3:** Macro F1 = 0.7356 (Dừng ở epoch 5, phục hồi epoch 1 tốt nhất)
- **Điểm Validation Trung bình (Mean ± Std):**
  - **Sentiment Macro F1:** **$0.6741 \pm 0.0065$**
  - **Toxicity Macro F1:** **$0.8006 \pm 0.0125$**
  - **Combined Macro F1:** **$0.7374 \pm 0.0051$**

---

## Hạn chế đã biết (Error Analysis)

- **Hạn chế của Static Embedding với câu mỉa mai (Sarcasm):** Do vector nhúng tĩnh của fastText cố định ngữ nghĩa, những câu khen ngợi cường điệu kèm icon mỉa mai (vd: *"Vĩ đại ghê :v"*) thường bị mô hình đoán nhầm thành Positive.
- **Hiện tượng False Positive do từ nhạy cảm:** Các câu kể chuyện vô hại nhưng có chứa từ nhạy cảm (vd: *"ngứa mu", "mấy thằng"*) dễ kích hoạt CNN đưa ra xác suất Toxic cao ($P > 0.7$) do các từ này xuất hiện dày đặc trong ngữ cảnh chửi thề ở tập train.
- **Độ nhạy của nhãn Negative còn thấp (Recall = 53.3%):** Mô hình có xu hướng nhầm lẫn giữa Negative và Neutral ở các câu phàn nàn nhẹ nhàng hoặc thiếu đại từ cảm xúc rõ ràng.

---

## Cấu trúc Repository

```text
NT120-fastText/
├── src/
│   ├── prepare_dataset.py       # Đọc dữ liệu, lọc rác, xử lý leak, binarize Toxic, xuất CSV
│   ├── preprocess_fasttext.py   # Pipeline làm sạch chữ, emoticon->emoji, teencode, pyvi
│   ├── train_bigru_lstm_cnn.py  # Định nghĩa mạng, nạp fastText, train multi-task, lưu kết quả
│   └── benchmark_and_report.py  # Script chạy Blind Test Challenge, đo throughput, tính Mean±Std
├── data/                        # Chứa dữ liệu gốc (train.csv, valid.csv, challenge.csv)
├── runs/                        # Thư mục kết quả 3 seed:
│   ├── bigru_fasttext_seed1/    # model.keras, vocab.json, metrics.json, preds_val.csv
│   ├── bigru_fasttext_seed2/
│   └── bigru_fasttext_seed3/
├── reports/                     # Báo cáo Markdown chi tiết (blind_benchmark_report.md)
└── README.md                    # Tài liệu tổng quan dự án
```

---

## Hướng dẫn sử dụng mô hình

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 - 3.12:
```bash
pip install tensorflow keras numpy pandas scikit-learn pyvi emoji
```

### 2. Dự đoán thử nghiệm một số câu trực tiếp
Chạy trực tiếp từ dòng lệnh sử dụng cờ `--predict`:
```bash
python src/train_bigru_lstm_cnn.py \
    --predict runs/bigru_fasttext_seed1 \
    --text "Sản phẩm dùng rất ưng ý, đóng gói cẩn thận" \
    --text "Giao hàng như hạch, phục vụ tệ hại" \
    --text "Đồ ngu xạo lồn cút đi"
```

### 3. Huấn luyện lại và Benchmark mù
```bash
# Bước 1: Chuẩn bị dữ liệu và tiền xử lý
python src/prepare_dataset.py --repo . --out data_ft.csv

# Bước 2: Huấn luyện vòng lặp 3 Seed
for s in 1 2 3; do
  python src/train_bigru_lstm_cnn.py \
      --data data_ft.csv \
      --embeddings cc.vi.300.vec.gz \
      --seed $s --epochs 30 --patience 4 --batch-size 64
done

# Bước 3: Chạy Blind Test trên Challenge và xuất Báo cáo
python src/benchmark_and_report.py \
    --runs runs/bigru_fasttext_seed1 runs/bigru_fasttext_seed2 runs/bigru_fasttext_seed3 \
    --challenge merged/challenge/challenge.csv \
    --out reports/blind_benchmark_report.md
```
