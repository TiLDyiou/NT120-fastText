# Phân tích Lỗi Định tính & Đối chiếu Hai Thế hệ Pipeline (Phase 7)

> Dựa trên dự đoán thực tế của `seed_1` ([preds_challenge_seed1.csv](file:///D:/HK5/AI/DoAn/antigravity-nt120/local_train/reports/preds_challenge_seed1.csv)) và báo cáo chuẩn ([challenge_report.md](file:///D:/HK5/AI/DoAn/antigravity-nt120/local_train/reports/challenge_report.md)) trên 3,237 bình luận tập mù `challenge`.

---

## 1. Bức tranh Tổng thể & So sánh với Bản Cũ (Keras 4 lớp)

| Chỉ số trên `challenge` ($N = 3,237$) | Bản cũ v1 (Keras + `.vec` + `pyvi`) | **Bản mới v2 (PyTorch + `.bin` + Song song)** | Chênh lệch / Ý nghĩa |
| :--- | :---: | :---: | :--- |
| **Toxicity Macro-F1 (4 lớp — chính)** | `0.5731 ± 0.0531` | **`0.6593 ± 0.0037`** | **+0.0862 (+8.62 điểm %)**; độ lệch chuẩn giảm **~14 lần** (hội tụ cực ổn định) |
| **Sentiment Macro-F1 (3 lớp)** | `0.6423 ± 0.0079` | **`0.6439 ± 0.0067`** | Nhẹ hơn về biến thiên seed, đồng đều trên cả 3 seed |
| **Combined Macro-F1** | `0.6077 ± 0.0250` | **`0.6516 ± 0.0038`** | **+0.0439 (+4.39 điểm %)** |
| **QWK (Toxicity 4 mức)** | *(không đo)* | **`0.6755 ± 0.0021`** | Độ tương hợp có trọng số khoảng cách đạt mức khá cao |
| **ECE Sentiment (Seed 1)** | `0.2440` | **`0.0782`** | Giảm **> 3 lần** nhờ bỏ `balanced` thô và thêm Temperature Scaling $\bar{T}_{\text{sent}}=1.4695$ |
| **ECE Toxicity (Seed 1)** | `0.0776` | **`0.0203`** | Giảm **~3.8 lần** ($\bar{T}_{\text{tox}}=1.2857$) |

### Tại sao Pipeline mới thắng lớn ở Độ ổn định và Toxicity 4 lớp?
1. **Làm dịu trọng số lớp $\alpha = 0.5, \text{clip} = 10$:** Lớp 3 chỉ chiếm ~0.65% dữ liệu train (`192/29,474`). Bản cũ dùng `class_weight="balanced"` ($\alpha=1$) làm trọng số lớp 3 gấp ~135 lần lớp 0 → gây nổ gradient và khiến Seed 2 tụt còn `0.4993`. Bản mới giữ tỷ lệ phạt ~6.2× giúp cả 3 seed đạt `0.6568 – 0.6635`.
2. **Xử lý OOV thực sự bằng `cc.vi.300.bin`:** `challenge` có **1,079 token mới** chưa từng xuất hiện trong `train + valid`. Nhờ Dynamic Vocab Expansion tra subword trực tiếp từ `.bin` (thay vì gộp hết thành `<unk>` như bản `.vec`), mạng vẫn nhận được vector có nghĩa.
3. **Kiến trúc Song song + Masking chuẩn:** `Conv1D(k=2, 3)` bắt trực tiếp cụm âm tiết 2–3 tiếng ngay trên embedding, `Bi-GRU` dùng `pack_padded_sequence` không bị padding làm nhiễu chiều đọc ngược.

---

## 2. Phân tích 5 Nhóm Lỗi Thực tế từ `preds_challenge_seed1.csv`

### Nhóm 1: Bỏ sót Negative / Mỉa mai ngược (`Recall Negative = 38.4%` trên Seed 1)
Trên `valid`, `R2_17` đạt Sentiment F1 = `0.6993`, nhưng sang `challenge` giảm còn `0.6439` (`Precision Negative = 80.6%` rất cao, nhưng `Recall = 38.4%`).
- **Ví dụ thực tế:**
  - `"Cố mỉm cười nhưng lòng chẳng được vui :)))"` → Nhãn thật `(-1, 0)`, mô hình đoán `(1, 0)` với $P_{\text{pos}} = 0.955$.
  - `"Chỉ có thể là ở MỸ mới có cuộc cải nhau giữa tổng thống..."` → Nhãn thật `(-1, 0)`, mô hình đoán `(0, 0)` ($P_{\text{neu}} = 0.965$).
- **Nguyên nhân:**
  1. **Lệch phân phối nhãn giữa `train/valid` và `challenge` (đã cảnh báo trong `LABELING_GUIDELINES.md` §13):** `challenge` đã được duyệt lại toàn bộ theo quy tắc mới (mỉa mai/châm chọc → `-1`, nâng tỷ lệ `Negative` lên `33.1%`), trong khi `train` (`20.4%`) và `valid` (`21.1%`) chưa được duyệt lại theo quy tắc mới.
  2. **Giới hạn của Static Embedding:** Các từ `"mỉm cười"`, `"vui"`, `":)))"` mang vector tích cực tĩnh, át tín hiệu phủ định/mỉa mai. Đây là điểm mà **Thế hệ 3 (BaMiBERT)** với Self-Attention đa tầng sẽ khắc phục mạnh nhất.

### Nhóm 2: Viết HOA TOÀN BỘ (ALL CAPS) + Lỗi chính tả làm mất tín hiệu Toxic
- **Ví dụ thực tế:**
  - `"MÁY CON ĐIÊN NẦY SAO KHÔNG AI NHỐT LẠI ĐỂ LOẠN HẾT"` → Nhãn thật `(-1, 2)`, mô hình đoán `(0, 0)` ($P_{\text{t0}} = 0.998$, $P_{\text{neu}} = 0.974$).
- **Nguyên nhân:** Do giữ nguyên chữ hoa (`case-sensitive`) theo spec và gặp biến thể vùng miền/gõ sai (`MÁY` = mấy, `NẦY` = này), toàn bộ cụm `"MÁY CON ĐIÊN NẦY"` bị lệch xa các mẫu chữ thường trong `train` và từ điển Common Crawl.

### Nhóm 3: Báo nhầm Toxic (False Positive) do "bẫy từ khóa"
- **Ví dụ thực tế:**
  - `"Chính sách ngu dân ngày càng hiệu quả"` → Nhãn thật `(-1, 0)`, mô hình đoán `(-1, 2)` ($P_{T2} = 0.994$). Chữ `"ngu"` trong từ ghép `"ngu dân"` kích hoạt `Conv1D` dù câu chỉ phê bình chính sách.
  - `"tao méc mấy a dlv tụi bây phản động nè"` → Nhãn thật `(-1, 0)`, mô hình đoán `(-1, 3)` ($P_{T3} = 0.948$). Trong `train`, từ `"dlv"` thường gắn với mức sàn `T=3`, nên câu đùa giễu nhẹ (được quy định `(-1, 0)` trong Casebook §7) bị phạt oan thành `T=3`.
  - `"Xã tắc mang dâng cho ác quỷ... Đê hèn như một lũ điên"` → Nhãn thật `(-1, 0)` (thơ chính trị không tục), mô hình đoán `(-1, 2)` ($P_{T2} = 0.487$).

### Nhóm 4: Bỏ sót công kích ngầm / không có từ tục (`y_tox >= 2`, `pred_tox = 0`)
- **Ví dụ thực tế:**
  - `"Toàn bọn đầu trộm , đuôi cướp !"` → Thật `(-1, 2)`, đoán `(0, 0)` ($P_{T0} = 0.998$).
  - `"Mô phật. Không phải người nữa rồi. a di đà phật."` → Thật `(-1, 2)`, đoán `(0, 0)` ($P_{T0} = 0.997$).
  - `"Tương lai thanh niên Bến Tre đổi qua nghề ' Trộm ' hết."` → Thật `(-1, 3)` (miệt thị vùng miền ngầm), đoán `(0, 0)` ($P_{T0} = 0.994$).
  - `"May Cha H O an Tro. Cap. Ranh c. Di ngoi le. Nhieu chuyen..."` → Thật `(-1, 2)`, đoán `(0, 0)` ($P_{T0} = 0.997$) do viết không dấu + dấu chấm rải rác.
- **Nguyên nhân:** Khi câu xúc phạm/kỳ thị dùng thành ngữ hoặc suy luận ngữ nghĩa ngầm mà không chứa từ chửi thề quen thuộc, mạng dựa trên static embedding + Conv1D cục bộ rất khó nhận diện.

### Nhóm 5: Nhập nhằng ranh giới giữa `T1` (Thiếu lịch sự) và `T2` (Xúc phạm)
- **Ví dụ thực tế:**
  - `"Ây long bùi => lông buồi"` → Thật `(0, 1)` (casebook: nhắc/chơi chữ cảm thán không nhắm ai → T1), đoán `(-1, 2)`.
  - `"Nhìn mặt xanh xao mà thương bác thật sự ... nhắc đến lại DM NHUNG"` → Thật `(-1, 2)` (casebook: viết tắt nhưng nhắm thẳng người có tên → T2), đoán `(-1, 1)`.
  - `"Đm thằng ad xamlone"` → Thật `(-1, 2)`, đoán `(-1, 1)` (vì `"xamlone"` là biến thể né chữ ít gặp hơn từ gốc).
- **Nguyên nhân:** Việc phân biệt `T1` và `T2` phụ thuộc chặt vào *đối tượng bị nhắm tới* (có công kích cá nhân hay chỉ cảm thán buột miệng) và *mức độ lách luật của từ tục* — đòi hỏi hiểu cấu trúc câu sâu hơn ở các mô hình Transformer.

---

## 3. Đóng góp & Gợi ý cho So sánh với BaMiBERT (Thế hệ 3)
1. Pipeline v2 thiết lập **mức chuẩn (baseline) Deep Learning cổ điển rất vững**: **`Toxicity Macro-F1 = 0.6593 ± 0.0037`**, **`QWK = 0.6755`**, **`ECE Toxicity = 0.0172`**, và tốc độ suy luận **`~9,500 câu/s`** (Warm start trên RTX 3050).
2. Khi so sánh với **BaMiBERT**, các nhóm ca lỗi ở Mục 2 (ALL CAPS, mỉa mai ngược, phân biệt ngữ cảnh `"ngu dân"` vs `"thằng ngu"`, `"dlv"` đùa vs công kích, kỳ thị ngầm vùng miền) chính là thước đo định tính lý tưởng để chứng minh giá trị của cơ chế Self-Attention tiền huấn luyện.
