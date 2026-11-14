# Đặc tả Kỹ thuật Chi tiết: Pipeline fastText → Mạng Nơ-ron Đa nhiệm

```text
Phase 1: Dữ liệu, Làm sạch & Tách token
  ↓
Phase 2: Từ điển & Ma trận fastText (cc.vi.300.bin)
  ↓
Phase 3: Kiến trúc Mô hình Song song (Bi-GRU ‖ Conv1D) & Cơ chế Huấn luyện
  ↓
Phase 4: Tuning Siêu tham số 3 Vòng (trên valid) & Ablation Log
  ↓
Phase 5: Calibration (Temperature T) & Điều chỉnh β (trên valid + kiểm tra V1/V2)
  ↓
Phase 6: Train lại Mô hình Cuối cùng (trên train + valid, 3 seed)
  ↓
Phase 7: Đánh giá trên Tập mù Challenge (đúng 1 lần duy nhất)
```

---

## 0. Nguyên tắc Bất biến

1. **Mục tiêu cốt lõi**: Đo lường trọn vẹn năng lực biểu diễn của fastText tiếng Việt (`cc.vi.300.bin`) khi được khai thác tối đa, bao gồm cả khả năng fine-tune vector của các token đã biết trong quá trình huấn luyện và khả năng sinh vector từ subword cho các token mới gặp lúc dự đoán.
2. **Tập `challenge` là tập mù tuyệt đối**: Chỉ được chạm vào `challenge` đúng một lần duy nhất ở Phase 7 để chạy dự đoán và xuất báo cáo điểm số. Tuyệt đối không dùng dữ liệu `challenge` để dựng từ điển, tính độ dài cắt câu `max_len`, tính trọng số lớp (`class weights`), tìm hệ số $\beta$, tìm hệ số nhiệt độ $T$, hay thống kê tần suất token.
3. **Không gộp lớp**:
   - **Sentiment** luôn cố định **3 lớp**: `-1` (Negative), `0` (Neutral), `1` (Positive).
   - **Toxicity** luôn cố định **4 lớp**: `0`, `1`, `2`, `3` (tương ứng từ không độc hại đến mức độ độc hại cao nhất).
4. **Tính tái lập và minh bạch**: Mọi quy tắc chọn mô hình, quy tắc tie-break và ngưỡng giữ/bỏ tham số hậu xử lý đều được định nghĩa cứng trước khi chạy thực nghiệm.

---

## Phase 1: Dữ liệu, Làm sạch và Tách token

### 1.1. Đầu vào dữ liệu và ánh xạ nhãn

- **Ba tập dữ liệu đầu vào**:
  - Tập huấn luyện: `data/train/train.csv`
  - Tập kiểm định: `data/valid/valid.csv`
  - Tập đánh giá mù cuối cùng: `data/challenge/challenge.csv`
- **Cấu trúc cột trong mỗi file CSV**:
  - `Comment`: chuỗi văn bản gốc của bình luận.
  - `Sentiment`: nhãn cảm xúc gốc thuộc $\{-1, 0, 1\}$. Khi đưa vào hàm loss của PyTorch (`CrossEntropyLoss`), ánh xạ một-một sang chỉ số lớp $\{0, 1, 2\}$ theo công thức `y_sent_idx = Sentiment + 1`. Khi xuất báo cáo dự đoán, ánh xạ ngược về $\{-1, 0, 1\}$.
  - `Toxicity` (tương ứng cột `tox_level` trong thiết kế): nhãn mức độ độc hại thuộc $\{0, 1, 2, 3\}$, giữ nguyên giá trị làm chỉ số lớp `y_tox_idx ∈ {0, 1, 2, 3}`.

### 1.2. Làm sạch văn bản (Text Cleaning)

Áp dụng cùng một hàm `clean_text(line)` độc lập cho từng dòng trên cả ba tập `train`, `valid`, `challenge` theo đúng 4 bước tuần tự sau:

1. **Chuẩn hóa bảng mã Unicode về dạng `NFC`** (`unicodedata.normalize("NFC", text)`):
   - _Lý do kỹ thuật_: Trong Unicode, cùng một chữ cái tiếng Việt có dấu có thể được lưu dưới dạng dựng sẵn (`NFC`, 1 ký tự) hoặc dạng tổ hợp (`NFD`, ký tự gốc + ký tự dấu rời nhau). Vì fastText cắt subword (character n-grams) trực tiếp trên chuỗi ký tự Unicode, nếu không đưa hết về `NFC`, cùng một từ hiển thị giống hệt nhau trên màn hình sẽ bị băm thành các mảnh subword khác nhau và nhận hai vector khác nhau.
2. **Thay thế đường dẫn web bằng token `url`**:
   - Mọi chuỗi khớp mẫu đường link (`http://...`, `https://...`, `www....`) được thay bằng chuỗi cố định `url` (có khoảng trắng tách biệt nếu dính sát chữ khác).
3. **Chuẩn hóa khoảng trắng**:
   - Gộp các khoảng trắng liên tiếp (space, tab, newline) thành một khoảng trắng đơn `' '` và cắt bỏ khoảng trắng thừa ở hai đầu chuỗi (`strip()`).
4. **Loại bỏ dòng rỗng**:
   - Sau 3 bước trên, nếu chuỗi rỗng (`""`) thì loại bỏ dòng đó khỏi tập dữ liệu.

> **Ràng buộc nghiêm ngặt (Không làm gì ngoài 4 bước trên)**:
>
> - **Không** chuyển chữ hoa thành chữ thường (`lower()`).
> - **Không** tách từ ghép tiếng Việt bằng `pyvi` hay bất kỳ công cụ word segmentation nào.
> - **Không** chuẩn hóa teencode, không xóa hay dịch emoji/emoticon thành chữ.
> - **Không** thu gọn chữ cái lặp (như `đẹpppp`) và **không** can thiệp vào chữ viết lách luật (như `đ.m`, `l.ồ.n`).

### 1.3. Khử trùng lặp chống rò rỉ dữ liệu (Deduplication)

- **Khóa so khớp (`dedup_key`)**: chính là chuỗi văn bản của dòng đó ngay sau khi chạy xong hàm `clean_text` ở Mục 1.2.
- **Quy tắc khử trùng lặp**:
  1. **Nội bộ tập `train`**: Duyệt từ đầu đến cuối file `train.csv`; nếu gặp các dòng có cùng `dedup_key`, chỉ giữ lại dòng xuất hiện đầu tiên và xóa các dòng trùng phía sau.
  2. **Nội bộ tập `valid`**: Duyệt từ đầu đến cuối file `valid.csv`; nếu gặp các dòng có cùng `dedup_key`, chỉ giữ lại dòng xuất hiện đầu tiên.
  3. **Chống rò rỉ giữa `train` và `valid`**: Mọi dòng trong `valid` có `dedup_key` trùng với bất kỳ dòng nào trong `train` đều bị **xóa khỏi `valid`** (ưu tiên giữ dữ liệu ở `train`).
  4. **Đối với tập `challenge`**: Chỉ loại bỏ dòng rỗng sau bước làm sạch; **giữ nguyên 100%** các dòng còn lại (không khử trùng nội bộ và không xóa chéo với `train`/`valid` để giữ nguyên phân phối tự nhiên của tập mù).

### 1.4. Tách token (Tokenization)

- **Cơ sở lựa chọn**: Bộ embedding `cc.vi.300.bin` được huấn luyện trên kho văn bản tiếng Việt tách theo khoảng trắng (đơn vị cơ sở chủ yếu là âm tiết). Do đó pipeline giữ nguyên đơn vị âm tiết, còn việc kết hợp ngữ nghĩa của từ ghép nhiều âm tiết sẽ do nhánh `Conv1D` đa kernel và nhánh `Bi-GRU` của mạng nơ-ron đảm nhận.
- **Thuật toán tách token trên chuỗi đã làm sạch**:
  1. Tách chuỗi theo khoảng trắng đơn thành danh sách các cụm thô (`raw_chunks`).
  2. Với mỗi cụm thô $c$, kiểm tra theo thứ tự:
     - **Trường hợp 1 (Cụm toàn ký hiệu / emoji / dấu câu)**: Nếu $c$ không chứa bất kỳ chữ cái (Unicode Letter) hay chữ số (Unicode Number) nào (ví dụ: `:))`, `<3`, `=)))`, `...!!!`, `???`), giữ nguyên cả cụm $c$ thành **1 token duy nhất**.
     - **Trường hợp 2 (Cụm có chứa chữ cái hoặc chữ số)**:
       - Nếu ở **đầu** cụm $c$ có chuỗi ký tự dấu câu/ký hiệu liên tiếp (không phải chữ cái/chữ số), bóc toàn bộ chuỗi ký hiệu ở đầu đó ra thành **1 token riêng**.
       - Nếu ở **cuối** cụm $c$ có chuỗi ký tự dấu câu/ký hiệu liên tiếp (không phải chữ cái/chữ số), bóc toàn bộ chuỗi ký hiệu ở cuối đó ra thành **1 token riêng**.
       - Phần lõi ở giữa (bắt đầu và kết thúc bằng chữ cái hoặc chữ số) được giữ nguyên thành **1 token**, bảo toàn toàn bộ dấu câu nằm xen giữa các chữ cái (như `đ.m`, `l.ồ.n`, `v.v`).
- **Ví dụ minh họa**:
  - Đầu vào: `Đồ ngu ngục. đ.m =)))` $\rightarrow$ `["Đồ", "ngu", "ngục", ".", "đ.m", "=)))"]`
  - Đầu vào: `"Ngục..." (đ.m) buồn:((` $\rightarrow$ `["\"", "Ngục", "...\"", "(", "đ.m", ")", "buồn", ":(("]`
- **Hạn chế đã biết và chấp nhận**:
  - Khi người viết gõ thiếu khoảng trắng sau dấu chấm câu (ví dụ `ngục.mày`), dấu chấm nằm giữa hai chữ cái nên thuật toán không phân biệt được với từ lách luật (`đ.m`) và giữ `ngục.mày` thành một token. Nhờ cơ chế subword của `cc.vi.300.bin`, token dính này vẫn được băm thành các mảnh ký tự con của `ngục` và `mày` nên vẫn sinh được vector có nghĩa.

---

## Phase 2: Từ điển và Ma trận fastText

### 2.1. Nguồn biểu diễn và quy ước chỉ số

- **Bắt buộc sử dụng `cc.vi.300.bin`** (vector $D = 300$ chiều), tuyệt đối không dùng file `.vec`. File `.bin` lưu toàn bộ bảng trọng số subword n-grams của fastText nên tính được vector cho mọi chuỗi ký tự bất kỳ.
- **Quy ước bảng từ điển (`token_to_id`)**:
  - Chỉ số `0` được dành riêng cho token `<pad>`. Dòng thứ `0` của ma trận embedding là vector không $\mathbf{0} \in \mathbb{R}^{300}$ và luôn bị khóa không nhận gradient (`padding_idx = 0`).
  - Các token thực tế trong tập dữ liệu huấn luyện được đánh số liên tiếp từ `1` đến $|V|$.
  - **Không có token `<unk>`**: mọi token xuất hiện (dù hiếm hay viết sai chính tả) đều có vector riêng từ `.bin`.

### 2.2. Phạm vi dựng từ điển và tính `max_len` theo từng giai đoạn

Để đảm bảo không rò rỉ thông tin giữa các tập dữ liệu, từ điển và độ dài chuỗi chuẩn `max_len` được tính riêng biệt theo đúng phạm vi dữ liệu được phép nhìn thấy ở giai đoạn đó:

| Giai đoạn                                    | Tập dữ liệu được dùng để dựng từ điển $V$ và tính `max_len` | Xử lý câu dài / ngắn hơn `max_len`                                                                                                                                                  |
| :------------------------------------------- | :---------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 4 & 5 (Tuning & Calibration)**       | Chỉ tính trên tập **`train`**                               | `max_len` = phân vị thứ 95 (95th percentile, làm tròn nguyên) của số token mỗi câu trong `train`. Câu dài hơn `max_len` bị cắt bỏ phần đuôi; câu ngắn hơn được pad bằng `0` ở cuối. |
| **Phase 5.3 (Kiểm tra chuyển giao $\beta$)** | Tính trên tập **`train + V1`**                              | `max_len` = phân vị thứ 95 của số token mỗi câu trong `train + V1`.                                                                                                                 |
| **Phase 6 & 7 (Train lại cuối & Challenge)** | Tính trên tập **`train + valid`**                           | `max_len_full` = phân vị thứ 95 của số token mỗi câu trong `train + valid`. Tập `challenge` ở Phase 7 bắt buộc dùng `max_len_full` này.                                             |

### 2.3. Tối ưu bộ nhớ RAM khi làm việc với `cc.vi.300.bin` (~7 GB)

Do model `cc.vi.300.bin` chiếm khoảng 7 GB RAM khi nạp:

1. Tách riêng một script/bước tiền xử lý embedding chạy độc lập trước khi train.
2. Trong bước này, nạp `cc.vi.300.bin` đúng một lần, duyệt qua toàn bộ token duy nhất trong từ điển để lấy vector 300 chiều (`ft_model.get_word_vector(token)`), đóng gói thành ma trận `float32` kích thước $(|V| + 1, 300)$ lưu xuống ổ cứng (`.npy` / `.pt`) kèm file `token_to_id.json`, sau đó giải phóng hoàn toàn `cc.vi.300.bin` khỏi RAM.
3. **Cơ chế lưu trước (Pre-caching) cho các token mới của tập `valid` trong giai đoạn Tuning**:
   - Khi chấm điểm trên `valid` ở Phase 4 và Phase 5, những token xuất hiện trong `valid` nhưng chưa có trong từ điển `train` ($V_{\text{train}}$) phải được xử lý bằng cơ chế mở rộng vector gốc lúc dự đoán (Mục 3.3).
   - Vì `valid` không phải tập mù, để tránh việc mỗi lần đánh giá cấu hình lại phải nạp 7 GB của `.bin` vào RAM, trong lần nạp `.bin` ở bước tiền xử lý, cho phép tra trước vector gốc của tập token $V_{\text{valid}} \setminus V_{\text{train}}$ và lưu vào một file riêng (`valid_oov_vectors.npy` + `valid_oov_map.json`).
   - **Ràng buộc cô lập**: Trong suốt quá trình huấn luyện trên `train`, tầng `nn.Embedding` của mạng chỉ chứa $|V_{\text{train}}| + 1$ dòng (hoặc nếu ghép sẵn bảng mở rộng thì các dòng thuộc $V_{\text{valid}} \setminus V_{\text{train}}$ tuyệt đối không bao giờ được truy cập hay nhận gradient/weight decay trong lúc train). Chỉ khi chuyển sang `model.eval()` để dự đoán trên `valid`, các dòng vector gốc này mới được nối vào bảng tra cứu.

---

## Phase 3: Kiến trúc Mô hình và Cơ chế Huấn luyện

### 3.1. Kiến trúc Thân mạng Duy nhất: Song song (`Bi-GRU ‖ Multi-kernel Conv1D`)

Toàn bộ pipeline cố định **một kiến trúc thân mạng duy nhất** là kiến trúc **Song song (`parallel`)** kết hợp giữa một nhánh `Bi-GRU` và hai nhánh `Conv1D` (`kernel = 2` và `kernel = 3`) đặt song song trực tiếp ngay sau biểu diễn token:

- **Lý do thiết kế**:
  1. Do đầu vào fastText tách theo từng âm tiết đơn, hai nhánh `Conv1D` với `kernel_size = 2` và `kernel_size = 3` đóng vai trò kết hợp trực tiếp các cụm từ ghép tiếng Việt 2–3 âm tiết và bắt các mẫu từ chửi thề/cảm thán cục bộ ngay từ mặt phẳng embedding.
  2. Nhánh `Bi-GRU` đọc hai chiều trên toàn câu giúp nắm bắt phụ thuộc ngữ cảnh xa (như cấu trúc phủ định, chuyển ý hoặc mỉa mai).
  3. Đặt `Bi-GRU` và `Conv1D` song song rồi nối đặc trưng trước khi Pooling giúp đường truyền gradient từ hàm loss về tầng `Embedding` ngắn, trực tiếp và tránh hiện tượng suy giảm thông tin cục bộ so với việc xếp chồng nối tiếp nhiều tầng RNN (`Bi-GRU → LSTM → Conv1D`).

Gọi $H \in \{64, 128, 256\}$ là siêu tham số số `unit`, luồng xử lý chi tiết từ đầu vào đến đầu ra như sau:

```text
Đầu vào:
  - input_ids:  LongTensor (B, L)   với L = max_len
  - actual_len: LongTensor (B,)     độ dài thật của từng câu (1 <= actual_len <= L)
      │
      ▼
1. Tầng Embedding:
  - nn.Embedding(num_embeddings, 300, padding_idx=0) khởi tạo từ ma trận fastText
  - Đầu ra: (B, L, 300)
      │
      ▼
2. Dropout đầu vào:
  - nn.Dropout(p=dropout)
      │
      ▼
3. [Tùy chọn] Tầng chiếu Dense 300:
  - nn.Linear(300, 300) + nn.ReLU()
  - Áp dụng cùng một phép biến đổi cho mọi vị trí token: X_in (B, L, 300)
      │
      ├───────────────────────────────────────────┐
      ▼                                           ▼
4a. Nhánh Ngữ cảnh Toàn cục (Bi-GRU):       4b. Nhánh Cụm Âm tiết Cục bộ (Multi-kernel Conv1D):
  - pack_padded_sequence theo actual_len      - Nhân X_in với mask.unsqueeze(-1)
  - Bi-GRU(input=300, hidden=H, bidir=True)   - Nhánh K=2: F.pad(0, 1) -> Conv1D(300, H, k=2) + ReLU
  - pad_packed_sequence(..., total_length=L)  - Nhánh K=3: Conv1D(300, H, k=3, padding=1) + ReLU
  - Đầu ra H_rnn: (B, L, 2H)                  - Nhân từng nhánh với mask.unsqueeze(-1)
      │                                       - Đầu ra H_cnn2: (B, L, H), H_cnn3: (B, L, H)
      └─────────────────────┬─────────────────────┘
                            ▼
5. Hợp nhất Đặc trưng (Feature Concatenation):
  - H_seq = torch.cat([H_rnn, H_cnn2, H_cnn3], dim=-1) -> Kích thước (B, L, 4H)
                            │
                            ▼
6. Masked Pooling (Max + Avg):
  - Masked Max Pooling theo chiều L -> (B, 4H)
  - Masked Avg Pooling theo chiều L -> (B, 4H)
  - Nối (concat) hai vector -> (B, 8H)
                            │
                            ▼
7. Tầng ẩn phân loại chung:
  - nn.Dropout(p=dropout)
  - nn.Linear(8H, 64) + nn.ReLU() -> (B, 64)
      │
      ├──► Đầu ra Sentiment: nn.Linear(64, 3) -> logits_sent (B, 3)
      └──► Đầu ra Toxicity:  nn.Linear(64, 4) -> logits_tox  (B, 4)
```

- **Không đặt `Softmax` bên trong `forward()` của mạng**: Mô hình trả về trực tiếp `logits` để đảm bảo tính ổn định số học khi đưa vào hàm loss và khi thực hiện Temperature Scaling ở Phase 5.
- **Vai trò của tầng tùy chọn `Dense 300`**: Khi fine-tune embedding, các token có trong `train` bị dịch chuyển vector trong khi các token mới gặp lúc dự đoán vẫn giữ vector gốc từ `.bin`. Tầng `Dense 300` áp cùng một ma trận chiếu lên mọi vector nhằm giảm độ lệch không gian giữa hai nhóm token này. Giả thuyết này được kiểm chứng trực tiếp bằng điểm `valid` ở Vòng 1 của Phase 4.

### 3.2. Quy tắc Masking xuyên suốt các tầng

Mặt nạ độ dài thật `mask` được định nghĩa thống nhất từ `actual_len`:
$$\text{mask}[b, t] = \mathbb{I}(t < \text{actual\_len}[b]) \in \{\text{True}, \text{False}\}^{B \times L}$$
Tuyệt đối **không** sinh `mask` bằng cách kiểm tra `input_ids != 0` để tránh phụ thuộc ngầm vào giá trị chỉ số mà dựa trực tiếp vào độ dài thật đã ghi nhận.

1. **Masking tại nhánh `Bi-GRU`**:
   - Dùng `nn.utils.rnn.pack_padded_sequence(X_in, actual_len.cpu(), batch_first=True, enforce_sorted=False)` trước khi đưa vào `Bi-GRU`.
   - Sau `Bi-GRU`, mở gói bằng `nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=L)`.
   - Nhờ đóng gói theo `actual_len`, chiều đọc ngược (backward pass) của `Bi-GRU` bắt đầu chính xác từ token cuối cùng thật sự của câu thay vì đọc qua các vị trí padding `0`.
2. **Masking và giữ nguyên chiều dài $L$ tại nhánh `Conv1D` đa kernel `(2, 3)`**:
   - Trước khi đưa $X_{\text{in}} \in \mathbb{R}^{B \times L \times 300}$ vào hai nhánh `Conv1D`, nhân $X_{\text{in}}$ với `mask.unsqueeze(-1)` để đảm bảo mọi vị trí padding đều bằng `0.0`.
   - Để giữ nguyên chiều dài chuỗi bằng $L$:
     - Với nhánh `kernel_size = 2` (số chẵn): thực hiện pad bất đối xứng 1 cột `0` vào bên phải theo chiều thời gian (`F.pad(x_transpose, (0, 1))`) rồi áp `Conv1D(..., kernel_size=2, padding=0)`.
     - Với nhánh `kernel_size = 3` (số lẻ): áp trực tiếp `Conv1D(..., kernel_size=3, padding=1)`.
   - Sau khi qua `Conv1D + ReLU`, tiếp tục nhân kết quả của cả hai nhánh với `mask.unsqueeze(-1)` một lần nữa để xóa bỏ hoàn toàn phần giá trị khác không bị tràn từ các token cuối câu sang vùng padding do cửa sổ trượt và `bias` của lớp tích chập tạo ra.
3. **Masking tại tầng Pooling**:
   - **Masked Max Pooling**: Với tensor hợp nhất $H_{\text{seq}} \in \mathbb{R}^{B \times L \times 4H}$, gán giá trị `-1e9` (hoặc $-\infty$) vào mọi vị trí có `mask == False`, sau đó lấy $\max$ dọc theo chiều thời gian $L$.
   - **Masked Avg Pooling**: Nhân $H_{\text{seq}}$ với `mask.unsqueeze(-1)`, tính tổng dọc theo chiều $L$, rồi chia cho `actual_len.unsqueeze(-1).clamp(min=1)`.

### 3.3. Cơ chế xử lý Token mới lúc dự đoán (Dynamic Vocabulary Expansion)

Khi chạy dự đoán trên tập đánh giá (`valid` hoặc `challenge`):

1. Nếu gặp một token $w$ chưa tồn tại trong bảng `token_to_id` hiện hành:
   - Tra vector gốc $\mathbf{v}_w \in \mathbb{R}^{300}$ từ `cc.vi.300.bin` (hoặc từ file pre-cache đối với `valid`).
   - Cấp cho $w$ chỉ số mới $i_{\text{new}} = \text{current\_vocab\_size}$ trong `token_to_id`.
   - Nối thêm dòng vector gốc $\mathbf{v}_w^\top$ vào cuối ma trận trọng số của tầng `nn.Embedding`.
2. Nếu gặp lại token $w$ ở các câu tiếp theo, dùng lại ngay chỉ số $i_{\text{new}}$ đã lưu.
3. **Tính chất bất biến**:
   - Không có bất kỳ bước cập nhật trọng số (gradient update) nào diễn ra khi dự đoán.
   - Cơ chế này thuần túy là một bộ nhớ đệm (lookup cache): kết quả dự đoán của một câu hoàn toàn không phụ thuộc vào thứ tự các câu được đưa vào dự đoán trước hay sau nó.

### 3.4. Chế độ Huấn luyện Embedding Duy nhất: Đóng băng $F$ vòng đầu rồi mở khóa với Learning Rate nhỏ (`freeze_then_open`)

Toàn bộ pipeline sử dụng cố định một chiến lược huấn luyện tầng `Embedding` duy nhất theo hai giai đoạn trong suốt $E$ vòng huấn luyện ($E \in \{3, 5, 8\}$, $F \in \{1, 2\}$, luôn thỏa mãn $1 \le F < E$):

1. **Giai đoạn 1 — Khởi động thân mạng ($1 \le \text{epoch} \le F$)**:
   - Đóng băng hoàn toàn tầng `Embedding` (`requires_grad = False`), chỉ cập nhật trọng số của các tầng phía trên (`Dense 300`, `Bi-GRU`, `Conv1D`, `Linear`).
   - _Mục đích_: Ở những vòng đầu tiên, các tầng phía trên vừa được khởi tạo ngẫu nhiên nên tạo ra gradient rất lớn và nhiễu. Đóng băng `Embedding` trong $F$ vòng đầu giúp thân mạng hội tụ về vùng ổn định trước, tránh làm hỏng cấu trúc ngữ nghĩa sẵn có của vector `cc.vi.300.bin`.
2. **Giai đoạn 2 — Tinh chỉnh nhẹ Embedding ($F + 1 \le \text{epoch} \le E$)**:
   - Mở khóa tầng `Embedding` (`requires_grad = True`, riêng dòng `padding_idx = 0` luôn khóa ở $\mathbf{0}$).
   - Đặt learning rate của nhóm tham số `Embedding` nhỏ hơn phần thân mạng theo hệ số `emb_lr_factor`:
     $$\eta_{\text{emb}}(t) = \text{emb\_lr\_factor} \times \eta_{\text{backbone}}(t), \quad \text{emb\_lr\_factor} \in \{0.1, 0.2, 0.3\}$$
   - _Mục đích_: Cho phép các từ khóa quan trọng (từ chửi thề, từ lóng, emoticon, từ phủ định) dịch chuyển nhẹ để phân tách sắc nét các mức Sentiment và Toxicity, đồng thời giữ các vector đã fine-tune không trôi quá xa khỏi không gian gốc của `cc.vi.300.bin` — nhờ đó các token mới được tra vector gốc lúc dự đoán vẫn tương thích với không gian biểu diễn của mạng.

### 3.5. Chế độ Đa nhiệm (Multi-task) và Đơn nhiệm (Single-task)

- **Đa nhiệm (`multi_task`)**:
  - Huấn luyện một mạng duy nhất có phần thân chung và hai đầu ra tuyến tính (`logits_sent` 3 chiều, `logits_tox` 4 chiều), cập nhật trọng số bằng tổng hàm loss của hai nhiệm vụ.
- **Đơn nhiệm (`single_task`)**:
  - Huấn luyện **hai mạng độc lập hoàn toàn** có cùng kiến trúc thân mạng Song song và cùng bộ siêu tham số:
    - `Model_Sentiment`: chỉ có đầu ra Sentiment (`Linear(64, 3)`), huấn luyện bằng loss của Sentiment.
    - `Model_Toxicity`: chỉ có đầu ra Toxicity (`Linear(64, 4)`), huấn luyện bằng loss của Toxicity.
  - Khi chấm điểm chọn cấu hình, lấy dự đoán Sentiment từ `Model_Sentiment` và dự đoán Toxicity từ `Model_Toxicity` để tính `Combined Macro-F1`.
  - Khi tính tổng số tham số để xét tie-break ở Vòng 3, tổng số tham số của phương án Đơn nhiệm bằng tổng số tham số của cả hai mạng cộng lại.

### 3.6. Quy trình Tối ưu hóa, Trọng số lớp và Hàm Loss

- **Thuật toán tối ưu**: `AdamW` với `batch_size = 64`.
- **Cắt gradient (Gradient Clipping)**: Sau mỗi bước `loss.backward()`, áp dụng `clip_grad_norm_(model.parameters(), max_norm=1.0)`.
- **Lịch giảm Learning Rate**:
  - Dùng lịch Cosine Annealing (`CosineAnnealingLR`) giảm mượt từ `lr` ban đầu về `0` (hoặc `1e-6`) trong đúng $E$ vòng huấn luyện.
  - Lịch Cosine được tính theo chỉ số vòng toàn cục $t \in \{1, \dots, E\}$, kể cả khi tầng `Embedding` được mở khóa giữa chừng ở vòng $F + 1$.
- **Không dùng Early Stopping và không giảm Learning Rate theo điểm valid**:
  - Mô hình luôn chạy hết đúng $E$ vòng và xuất trọng số ở cuối vòng $E$.
  - _Lý do_: Khi train lại lần cuối trên gộp `train + valid` ở Phase 6, không còn tập validation nào để canh thời điểm dừng sớm. Việc coi $E$ là một siêu tham số cố định giúp quá trình train lại ở Phase 6 đi theo đúng quỹ đạo tối ưu như ở Phase 4.
- **Cố định Seed**: Đặt seed cho `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all` trước khi khởi tạo mô hình và `DataLoader`.
- **Công thức Trọng số lớp (Class Weights) cho Toxicity**:
  - Nhiệm vụ Sentiment có phân phối 3 lớp tương đối cân bằng nên mặc định không dùng trọng số lớp ($w_c^{\text{sent}} = 1$).
  - Nhiệm vụ Toxicity có lớp 3 rất hiếm; nếu dùng trọng số tỷ lệ nghịch trực tiếp ($\alpha = 1$), trọng số của lớp 3 sẽ lớn gấp cả trăm lần lớp 0, làm khuếch đại nhiễu gradient. Do đó, trọng số lớp $k \in \{0, 1, 2, 3\}$ được tính bằng công thức làm dịu có giới hạn trần:
    $$w_k = \min\left( \left( \frac{N}{4 \cdot n_k} \right)^\alpha, \, \text{clip\_max} \right)$$
    trong đó $N$ là tổng số mẫu của tập huấn luyện hiện hành, $n_k$ là số mẫu thuộc lớp Toxicity $k$, $\alpha \in \{0, 0.5\}$ và $\text{clip\_max}$ là mức trần trọng số (khi $\alpha = 0$ thì $w_k = 1$ với mọi $k$).
- **Ba lựa chọn Hàm Loss**:
  1. **Cộng thường (`weighted_ce`)**:
     $$\mathcal{L} = \mathcal{L}_{\text{CE}}^{\text{sent}} + w_{\text{tox}} \cdot \mathcal{L}_{\text{CE}}^{\text{tox}}(w_k)$$
     trong đó $\mathcal{L}_{\text{CE}}^{\text{tox}}(w_k)$ là Cross-Entropy có trọng số lớp $w_k$, và $w_{\text{tox}}$ là hệ số cân bằng nhiệm vụ.
  2. **Focal Loss cho Toxicity (`focal_tox`)**:
     - Giảm đóng góp của các mẫu dễ (có xác suất dự đoán đúng $p_{i, y_i}$ đã cao) bằng hệ số điều biến $(1 - p_{i, y_i})^\gamma$, kết hợp với trọng số lớp $w_k$:
       $$\mathcal{L}_{\text{Focal}}^{\text{tox}} = \frac{1}{B} \sum_{i=1}^{B} w_{y_i} (1 - p_{i, y_i})^\gamma \left( -\log p_{i, y_i} \right)$$
       $$\mathcal{L} = \mathcal{L}_{\text{CE}}^{\text{sent}} + w_{\text{tox}} \cdot \mathcal{L}_{\text{Focal}}^{\text{tox}}$$
     - _Hệ quả kỹ thuật_: Focal Loss làm biến dạng phân phối xác suất đầu ra (thường đẩy xác suất về gần mức trung bình, làm tăng ECE), do đó ảnh hưởng trực tiếp đến bước Temperature Scaling và tìm $\beta$ ở Phase 5.
  3. **Uncertainty Weighting (`uncertainty`)**:
     - Đặt hai tham số vô hướng học được $s_{\text{sent}}, s_{\text{tox}} \in \mathbb{R}$ (khởi tạo bằng `0.0`, biểu diễn $\log \sigma^2$) vào optimizer để mạng tự học tỷ lệ cân bằng giữa hai loss mà không dùng siêu tham số $w_{\text{tox}}$:
       $$\mathcal{L} = \exp(-s_{\text{sent}}) \cdot \mathcal{L}_{\text{CE}}^{\text{sent}} + \frac{1}{2} s_{\text{sent}} + \exp(-s_{\text{tox}}) \cdot \mathcal{L}_{\text{CE}}^{\text{tox}}(w_k) + \frac{1}{2} s_{\text{tox}}$$
     - _Lưu ý_: Công thức này vốn được dẫn xuất cho bài toán hồi quy Gaussian, nên khi áp vào bài toán phân loại chỉ là phép xấp xỉ.

---

## Phase 4: Tuning Siêu tham số trên tập `valid` (3 Vòng)

### 4.1. Thước đo chọn cấu hình (Selection Metric)

Mọi cấu hình trong Phase 4 đều được chấm điểm trên tập `valid` ở cuối vòng huấn luyện thứ $E$, sử dụng nhãn dự đoán trực tiếp $\arg\max$ từ logit của mạng (**chưa** áp hệ số $\beta$):
$$\text{Combined Macro-F1} = \frac{\text{Macro-F1}_{\text{Sentiment (3 lớp)}} + \text{Macro-F1}_{\text{Toxicity (4 lớp)}}}{2}$$

### 4.2. Vòng 1: Chọn khung Cơ chế (1 Seed cố định `seed = 1`)

1. **Thiết lập cấu hình mặc định gốc (`C0`) trước khi chạy**:
   - Kiến trúc thân mạng: cố định `Song song` (`Bi-GRU ‖ Conv1D kernels (2, 3)`)
   - Chế độ Embedding: cố định `freeze_then_open` (mặc định ở Vòng 1: đóng băng $F = 2$ vòng đầu rồi mở với `emb_lr_factor = 0.1`)
   - Tầng `Dense 300`: `Không` (`use_dense300 = False`)
   - Chế độ nhiệm vụ: `Đa nhiệm` (`multi_task`)
   - Hàm loss: `Cộng thường` (`weighted_ce`), $w_{\text{tox}} = 1.0$, $\alpha = 0.5$, $\text{clip\_max} = 10.0$
   - Các siêu tham số cơ sở: `lr = 5e-4`, `E = 5`, `dropout = 0.3`, `H = 128`, `weight_decay = 1e-4`.
2. **Danh sách 5 cấu hình chạy ở Vòng 1 (đổi từng yếu tố một so với `C0`)**:

| Mã cấu hình | Nhóm yếu tố khảo sát   | Thay đổi duy nhất so với `C0`                                   |
| :---------- | :--------------------- | :-------------------------------------------------------------- |
| `C0`        | Mặc định (Baseline)    | Giữ nguyên toàn bộ thiết lập mặc định gốc                       |
| `C1a`       | Chế độ Embedding       | Đổi sang `Mở ngay` từ vòng 1 (`open`)                           |
| `C1b`       | Chế độ Embedding       | Đổi sang `Đóng băng suốt` (`frozen`)                            |
| `C2`        | Tầng chiếu Dense 300   | Đổi sang `Có Dense 300` (`use_dense300 = True`)                 |
| `C3`        | Đa nhiệm vs. Đơn nhiệm | Đổi sang `Đơn nhiệm` (`single_task` — huấn luyện 2 mạng riêng)  |
| `C4a`       | Hàm Loss               | Đổi sang `Focal cho Toxicity` (`focal_tox`, với $\gamma = 2.0$) |
| `C4b`       | Hàm Loss               | Đổi sang `Uncertainty Weighting` (`uncertainty`)                |
| `C1`        | Tầng chiếu Dense 300   | Đổi sang `Có Dense 300` (`use_dense300 = True`)                 |
| `C2`        | Đa nhiệm vs. Đơn nhiệm | Đổi sang `Đơn nhiệm` (`single_task` — huấn luyện 2 mạng riêng)  |
| `C3a`       | Hàm Loss               | Đổi sang `Focal cho Toxicity` (`focal_tox`, với $\gamma = 2.0$) |
| `C3b`       | Hàm Loss               | Đổi sang `Uncertainty Weighting` (`uncertainty`)                |

3. **Quy tắc chốt khung sau Vòng 1 để chuyển sang Vòng 2**:
   - Xét 3 nhóm yếu tố độc lập: (1) Dense 300, (2) Đa/Đơn nhiệm, (3) Hàm Loss. Ở mỗi nhóm, xác định lựa chọn mang lại điểm `Combined Macro-F1` cao nhất (nếu không lựa chọn nào vượt `C0` ở nhóm đó thì giữ nguyên lựa chọn của `C0`).
   - Ghép 3 lựa chọn tốt nhất của 3 nhóm lại thành cấu hình tổ hợp `C_comb`.
   - Nếu `C_comb` khác với cả 5 cấu hình đã chạy, thực hiện chạy kiểm tra thêm đúng 1 lần cho `C_comb`:
     - Nếu `Score(C_comb)` $\ge$ điểm cao nhất trong 5 cấu hình đầu $\rightarrow$ chọn khung của `C_comb` đưa sang Vòng 2.
     - Nếu `Score(C_comb)` $<$ điểm cao nhất trong 5 cấu hình đầu (do tương tác bất lợi giữa các yếu tố) $\rightarrow$ chọn khung của cấu hình đơn lẻ có điểm cao nhất trong 5 lần chạy đầu đưa sang Vòng 2.

### 4.3. Vòng 2: Tinh chỉnh Siêu tham số Chi tiết (Random Search 20–30 cấu hình, `seed = 1`)

Giữ cố định khung cơ chế đã chốt ở Vòng 1, lấy mẫu ngẫu nhiên **20 đến 30 cấu hình** không trùng lặp từ không gian lưới siêu tham số sau:

| Siêu tham số                        | Tập giá trị ứng viên                      | Điều kiện áp dụng                                            |
| :---------------------------------- | :---------------------------------------- | :----------------------------------------------------------- |
| `lr`                                | `{1e-3, 5e-4, 2e-4}`                      | Mọi cấu hình                                                 |
| `E` (tổng số vòng)                  | `{2, 3, 5, 8}`                            | Mọi cấu hình (ràng buộc `E > F` nếu dùng `freeze_then_open`) |
| `dropout`                           | `{0.2, 0.3, 0.5}`                         | Mọi cấu hình                                                 |
| `H` (số unit)                       | `{64, 128, 256}`                          | Mọi cấu hình                                                 |
| `weight_decay`                      | `{0.0, 1e-4, 1e-2}`                       | Mọi cấu hình                                                 |
| `alpha` ($\alpha$)                  | `{0.0, 0.5}`                              | Mọi cấu hình                                                 |
| `clip_max` (trần $w_k$)             | `{5.0, 10.0, None}` (`None` = không trần) | Khi $\alpha = 0.5$                                           |
| `w_tox`                             | `{1.0, 2.0}`                              | Khi không dùng `Uncertainty Weighting`                       |
| `gamma` ($\gamma$)                  | `{1.0, 2.0}`                              | Chỉ khi khung dùng `Focal Loss`                              |
| `F` (số vòng đóng băng)             | `{1, 2}`                                  | Chỉ khi khung dùng `freeze_then_open` (bắt buộc $F < E$)     |
| `emb_lr_factor`                     | `{0.1, 0.2, 0.3}`                         | Chỉ khi khung dùng `freeze_then_open`                        |
| luônđảmbảo $$vớimọi$F \in \{1, 2\}$ |
| `F` (số vòng đóng băng đầu)         | `{1, 2}`                                  | Mọi cấu hình                                                 |
| `emb_lr_factor`                     | `{0.1, 0.2, 0.3}`                         | Mọi cấu hình                                                 |

### 4.4. Vòng 3: Xác nhận Đa Seed và Quy tắc Chọn Cấu hình Chiến thắng

1. Lấy **Top 3 cấu hình** có điểm `Combined Macro-F1` cao nhất từ toàn bộ nhật ký Vòng 1 + Vòng 2.
2. Huấn luyện và đánh giá lại mỗi cấu hình trong Top 3 với **3 seed độc lập** (`seed ∈ {1, 2, 3}`).
3. **Quy tắc chọn cấu hình cuối cùng (viết cứng trước khi chạy)**:
   - Với mỗi cấu hình $i \in \{1, 2, 3\}$, tính trung bình $\mu_i$ và độ lệch chuẩn mẫu $\sigma_i$ của `Combined Macro-F1` qua 3 seed. Sắp xếp 3 cấu hình theo thứ tự giảm dần của $\mu$ (gọi cấu hình đứng đầu là $A$, cấu hình so sánh là $B$).
   - **Kiểm tra 1 — Nhiễu huấn luyện giữa các seed (Seed Std Check)**:
     - Nếu $\mu_A - \mu_B \le \max(\sigma_A, \sigma_B)$, hai cấu hình $A$ và $B$ được coi là nằm trong phạm vi nhiễu seed.
   - **Kiểm tra 2 — Nhiễu mẫu hiếm trên `valid` (Paired Bootstrap Check)**:
     - Do lớp Toxicity 3 có rất ít mẫu trong `valid`, thực hiện lấy mẫu lại có hoàn lại (bootstrap) $B_{\text{iter}} = 1000$ lần trên tập các dòng của `valid`.
     - Ở mỗi lần lấy mẫu $b \in \{1, \dots, 1000\}$, tính điểm `Combined Macro-F1` trung bình 3 seed của $A$ và $B$ trên cùng mẫu bootstrap đó, thu được hiệu số $\Delta^{(b)} = \text{Score}_A^{(b)} - \text{Score}_B^{(b)}$.
     - Xác định khoảng tin cậy phân vị 95% ($\text{CI}_{95\%} = [\text{percentile}_{2.5}(\Delta), \text{percentile}_{97.5}(\Delta)]$). Nếu $0 \in \text{CI}_{95\%}$, hai cấu hình $A$ và $B$ được coi là ngang nhau về mặt thống kê trên dữ liệu `valid`.
   - **Quy tắc Tie-break (Ưu tiên cấu hình đơn giản hơn)**:
     - Nếu $A$ và $B$ được coi là ngang nhau theo Kiểm tra 1 **hoặc** Kiểm tra 2, bắt buộc chọn cấu hình có **tổng số tham số học được ít hơn** làm cấu hình chiến thắng $C^*$.

### 4.5. Nhật ký Thực nghiệm và Bảng Ablation

- Mọi lần chạy ở Vòng 1, Vòng 2, Vòng 3 (ước tính tổng cộng 35–45 lần huấn luyện) phải được ghi tự động vào file nhật ký `runs/tuning_log.jsonl` (và xuất ra bảng tổng hợp `reports/ablation_table.csv`), lưu đầy đủ: mã cấu hình, vòng chạy, seed, toàn bộ siêu tham số, số lượng tham số mô hình, thời gian huấn luyện, `Macro-F1 Sentiment`, `Macro-F1 Toxicity`, và `Combined Macro-F1`.

---

## Phase 5: Hiệu chuẩn Xác suất và Điều chỉnh Sau dự đoán (trên `valid`)

Phase này sử dụng trực tiếp bộ 3 mô hình đã huấn luyện trên `train` (tương ứng 3 seed của cấu hình chiến thắng $C^*$ ở Vòng 3).

### 5.1. Temperature Scaling và Báo cáo ECE

1. **Tối ưu hệ số nhiệt độ $T$**:
   - Với mỗi seed $s \in \{1, 2, 3\}$, trích xuất toàn bộ logit dự đoán của mô hình trên tập `valid`: $Z_{\text{sent}}^{(s)} \in \mathbb{R}^{N_{\text{val}} \times 3}$ và $Z_{\text{tox}}^{(s)} \in \mathbb{R}^{N_{\text{val}} \times 4}$.
   - Tìm hai hệ số vô hướng dương $T_{\text{sent}}^{(s)} > 0$ và $T_{\text{tox}}^{(s)} > 0$ độc lập cho hai đầu ra bằng cách tối thiểu hóa hàm Negative Log-Likelihood (Cross-Entropy Loss không trọng số) trên `valid`:
     $$T_{\text{task}}^{(s)} = \arg\min_{T > 0} \, \text{NLL}\left( \text{softmax}\left( \frac{Z_{\text{task}}^{(s)}}{T} \right), \, Y_{\text{task}}^{\text{val}} \right)$$
   - Lấy trung bình qua 3 seed để thu được bộ nhiệt độ cố định $\bar{T}_{\text{sent}} = \frac{1}{3}\sum_{s=1}^3 T_{\text{sent}}^{(s)}$ và $\bar{T}_{\text{tox}} = \frac{1}{3}\sum_{s=1}^3 T_{\text{tox}}^{(s)}$.
2. **Tính và báo cáo ECE (Expected Calibration Error)**:
   - Chia đoạn độ tự tin $[0, 1]$ thành $M = 15$ bin đều nhau.
   - Tính ECE (Mean ± Std qua 3 seed) cho cả Sentiment và Toxicity tại hai thời điểm: **trước** khi chia cho $T$ ($T = 1$) và **sau** khi chia cho $T$ (**trước khi** nhân hệ số $\beta$).
   - _Tính chất_: Phép chia logit cho $T > 0$ là phép biến đổi đơn điệu nghiêm ngặt nên không làm đổi kết quả $\arg\max$, do đó `Macro-F1` trước và sau khi áp $T$ giữ nguyên tuyệt đối; chỉ có độ khớp của xác suất dự đoán (ECE và NLL) được cải thiện.

### 5.2. Điều chỉnh Xác suất theo Tần suất lớp bằng Hệ số $\beta$ (Chỉ áp cho Toxicity)

1. **Định nghĩa trọng số điều chỉnh $\tau_k(\beta)$**:
   - Gọi $f_k = \frac{n_k}{N}$ là tần suất xuất hiện của lớp Toxicity $k \in \{0, 1, 2, 3\}$ trong tập huấn luyện.
   - Với tham số điều chỉnh $\beta \ge 0$, hệ số nhân cho lớp $k$ là:
     $$\tau_k(\beta) = \left( \frac{f_0}{f_k} \right)^\beta$$
   - Khi dự đoán từ xác suất đã hiệu chuẩn $p_{i, k} = \text{softmax}(z_{i} / T)_k$, nhãn Toxicity được chọn theo quy tắc:
     $$\hat{y}_{i, \text{tox}} = \arg\max_{k \in \{0, 1, 2, 3\}} \left( \tau_k(\beta) \cdot p_{i, k} \right)$$
2. **Lý do thiết kế**:
   - Thay vì tìm 4 ngưỡng rời rạc cho 4 lớp (rất dễ overfit vào vài mẫu hiếm của lớp 3 trong `valid`), công thức trên chỉ dùng **1 tham số tự do duy nhất là $\beta$**, trong khi mỗi lớp $k$ vẫn nhận một hệ số bù $\tau_k$ riêng biệt tỷ lệ với độ hiếm thực tế trên tập `train` và không lớp nào bị gộp.
3. **Tìm $\beta^\*$ trên toàn bộ `valid`**:
   - Duyệt lưới $\beta \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ ($\beta = 0.0$ tương ứng với $\tau_k = 1$, tức không điều chỉnh).
   - Chọn giá trị $\beta^*$ mang lại điểm **`Macro-F1 Toxicity` (4 lớp) trung bình qua 3 seed** cao nhất trên tập `valid`.

### 5.3. Kiểm tra Khả năng Chuyển giao của $\beta$ sang Mô hình Train lại (`V1` / `V2` Check)

Vì ở Phase 6 mô hình cuối sẽ được train trên gộp `train + valid` (đã nhìn thấy dữ liệu `valid`), nếu $\beta^* > 0$, bắt buộc thực hiện phép kiểm tra sau để xác nhận việc áp $\beta$ vẫn có lợi khi mô hình được học thêm dữ liệu:

1. **Chia tách phân tầng**: Chia tập `valid` thành hai tập con `V1` (50%) và `V2` (50%) theo phương pháp phân tầng (`stratified split` với seed cố định) dựa trên nhãn `Toxicity` (0..3).
2. **Tìm $\beta\_{\text{V1}}$ trên `V1`**: Dùng bộ 3 mô hình train-only (đã có từ Vòng 3, chỉ học trên `train`) để tìm $\beta_{\text{V1}} \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ tối ưu hóa `Macro-F1 Toxicity` trung bình 3 seed trên `V1` (sử dụng $T$ học trên `V1`).
3. **Huấn luyện bộ mô hình kiểm chứng trên `train + V1`**:
   - Dựng từ điển và `max_len` từ `train + V1`, tính lại $w_k$ và $f_k$ trên `train + V1`.
   - Huấn luyện bộ 3 mô hình mới (cùng cấu hình $C^*$, cùng 3 seed `1, 2, 3`) trên `train + V1` trong đúng $E$ vòng.
4. **Đo mức tăng điểm trên tập kiểm thử độc lập `V2`**:
   - Với bộ mô hình `train`-only, tính độ chênh lệch `Macro-F1 Toxicity` trung bình 3 seed trên `V2` khi bật $\beta_{\text{V1}}$ so với khi tắt ($\beta = 0$):
     $$\Delta_{\text{train}} = \overline{\text{F1}}_{\text{tox}}(\text{model}_{\text{train}}, \text{V2}, \beta_{\text{V1}}) - \overline{\text{F1}}_{\text{tox}}(\text{model}_{\text{train}}, \text{V2}, \beta = 0)$$
   - Với bộ mô hình `train + V1`, tính độ chênh lệch tương tự trên `V2`:
     $$\Delta_{\text{train+V1}} = \overline{\text{F1}}_{\text{tox}}(\text{model}_{\text{train+V1}}, \text{V2}, \beta_{\text{V1}}) - \overline{\text{F1}}_{\text{tox}}(\text{model}_{\text{train+V1}}, \text{V2}, \beta = 0)$$
5. **Quy tắc quyết định giữ hay hủy $\beta^\*$ (viết cứng trước khi chạy)**:
   - Nếu đồng thời thỏa mãn cả hai điều kiện:
     $$\Delta_{\text{train+V1}} \ge 0 \quad \text{và} \quad \Delta_{\text{train+V1}} \ge 0.5 \times \Delta_{\text{train}}$$
     $\Rightarrow$ **Giữ nguyên** $\beta_{\text{final}} = \beta^*$ (giá trị tìm được ở Mục 5.2).
   - Nếu vi phạm một trong hai điều kiện trên:
     $\Rightarrow$ **Hủy điều chỉnh sau dự đoán**, đặt $\beta_{\text{final}} = 0.0$ (chỉ dựa vào trọng số lớp $w_k$ lúc huấn luyện).
6. **Lưu ý báo cáo**:
   - Chỉ đọc kết quả trên chỉ số `Macro-F1 Toxicity` tổng 4 lớp, không rút kết luận riêng lẻ trên lớp 3.
   - Ghi nhận rõ trong báo cáo: do `V2` có rất ít mẫu lớp hiếm, phép kiểm tra này có độ mạnh thống kê thấp; việc vượt qua kiểm tra chỉ xác nhận chưa có bằng chứng cho thấy $\beta$ phản tác dụng khi gộp dữ liệu, và việc giữ nguyên $(\bar{T}, \beta_{\text{final}})$ sang Phase 6 vẫn là một phép xấp xỉ có kiểm soát.

---

## Phase 6: Huấn luyện lại Mô hình Cuối cùng (trên `train + valid`)

1. **Hợp nhất dữ liệu**:
   - Gộp toàn bộ tập `train` và tập `valid` (đã làm sạch và khử trùng lặp ở Phase 1) thành tập huấn luyện cuối cùng `full_train`.
2. **Dựng lại toàn bộ thống kê từ `full_train`**:
   - Dựng bảng từ điển $V_{\text{full}}$ và trích xuất ma trận embedding gốc $E_{\text{full}} \in \mathbb{R}^{(|V_{\text{full}}| + 1) \times 300}$ từ `cc.vi.300.bin`.
   - Tính lại `max_len_full` bằng phân vị thứ 95 của độ dài câu trên `full_train`.
   - Tính lại trọng số lớp $w_k^{\text{full}}$ và tần suất lớp $f_k^{\text{full}}$ từ phân phối nhãn Toxicity của `full_train`.
3. **Huấn luyện 3 Seed độc lập**:
   - Huấn luyện từ đầu (from scratch) cấu hình chiến thắng $C^*$ trên `full_train` trong đúng $E$ vòng với 3 seed cố định: `seed = 1`, `seed = 2`, `seed = 3`.
   - **Tuyệt đối không gộp thành Ensemble**: mỗi seed tạo ra một mô hình hoàn chỉnh và được đánh giá độc lập ở Phase 7 để tính `Mean ± Std`.
4. **Đóng gói bộ tham số hậu xử lý và lưu trữ Artifact**:
   - Gắn cố định các hằng số $(\bar{T}_{\text{sent}}, \bar{T}_{\text{tox}}, \beta_{\text{final}})$ đã chốt ở Phase 5 vào cấu hình suy luận của cả 3 mô hình (trong đó trọng số $\tau_k(\beta_{\text{final}}) = (f_0^{\text{full}} / f_k^{\text{full}})^{\beta_{\text{final}}}$).
   - Với mỗi seed $s \in \{1, 2, 3\}$, lưu đầy đủ vào thư mục `runs/final_models/seed_{s}/`:
     - Trọng số mô hình (`model_weights.pt`)
     - Bảng từ điển (`token_to_id.json`) và ma trận embedding sau khi train (`embedding_matrix.npy`)
     - File cấu hình hoàn chỉnh (`config.json` chứa kiến trúc, siêu tham số, `max_len_full`, $\bar{T}_{\text{sent}}$, $\bar{T}_{\text{tox}}$, $\beta_{\text{final}}$, $\tau_k$)
     - Nhật ký loss theo từng vòng (`train_history.json`).

---

## Phase 7: Đánh giá trên Tập mù `challenge` (Đúng một lần duy nhất)

### 7.1. Quy trình Suy luận Chuẩn trên từng câu của `challenge`

Với mỗi mô hình trong 3 seed cuối cùng (`seed = 1, 2, 3`), thực hiện dự đoán trên tập `challenge` theo đúng luồng sau:

1. **Làm sạch & Tách token**: Chạy `clean_text` (Mục 1.2) và tách token (Mục 1.4).
2. **Ánh xạ chỉ số & Xử lý Token mới**:
   - Với mỗi token trong câu, tra trong bảng `token_to_id` hiện tại của mô hình.
   - Nếu token chưa có trong từ điển: tra vector gốc 300 chiều từ `cc.vi.300.bin`, thêm một dòng vector gốc đó vào cuối bảng `Embedding` và ghi nhớ chỉ số mới vào `token_to_id` cho các lần gặp sau.
3. **Cắt / Pad và tạo Mask**:
   - Dùng đúng `max_len_full` của Phase 6 để cắt/pad chuỗi, ghi nhận `actual_len` và tạo `mask`.
4. **Dự đoán và Hậu xử lý**:
   - Chạy `forward()` để lấy logit $z_{\text{sent}} \in \mathbb{R}^3$ và $z_{\text{tox}} \in \mathbb{R}^4$.
   - Tính xác suất hiệu chuẩn:
     $$p_{\text{sent}} = \text{softmax}\left( \frac{z_{\text{sent}}}{\bar{T}_{\text{sent}}} \right), \quad p_{\text{tox}} = \text{softmax}\left( \frac{z_{\text{tox}}}{\bar{T}_{\text{tox}}} \right)$$
   - Chọn nhãn Sentiment: $\hat{y}_{\text{sent}} = \arg\max_{c \in \{0, 1, 2\}} p_{\text{sent}, c} - 1 \in \{-1, 0, 1\}$.
   - Chọn nhãn Toxicity sau khi nhân $\tau_k(\beta_{\text{final}})$:
     $$\hat{y}_{\text{tox}} = \arg\max_{k \in \{0, 1, 2, 3\}} \left( \tau_k(\beta_{\text{final}}) \cdot p_{\text{tox}, k} \right)$$

> **Ràng buộc chống rò rỉ**:
>
> - **Được phép**: Tra `.bin` và nối thêm vector gốc vào bảng tra cứu lúc dự đoán.
> - **Cấm tuyệt đối**: Huấn luyện/cập nhật trọng số trên các dòng mới, hoặc tính lại `max_len`, trọng số lớp $w_k$, $\beta$, $T$, hay tần suất lớp $f_k$ từ dữ liệu `challenge`.

### 7.2. Hệ thống Chỉ số Báo cáo Bắt buộc

Mọi chỉ số điểm số dưới đây đều được tính riêng trên từng seed ($s = 1, 2, 3$) và báo cáo dưới dạng **`Mean ± Std` của 3 seed**:

1. **Nhiệm vụ Sentiment (3 lớp: `-1`, `0`, `1`)**:
   - `Macro-F1` (3 lớp).
   - `F1-score` riêng của từng lớp (`Negative`, `Neutral`, `Positive`) kèm số lượng mẫu thật (`support`).
   - `Confusion matrix` kích thước $3 \times 3$.
2. **Nhiệm vụ Toxicity (4 lớp: `0`, `1`, `2`, `3`)**:
   - **`Macro-F1` (4 lớp)** — chỉ số đánh giá chính của bài toán.
   - `F1-score` riêng của từng lớp (`Lớp 0`, `Lớp 1`, `Lớp 2`, `Lớp 3`) kèm `support`.
   - `Confusion matrix` kích thước $4 \times 4$.
   - **Khoảng tin cậy Bootstrap 95%**: Lấy mẫu bootstrap $1000$ lần trên các dòng của tập `challenge`, tính khoảng tin cậy phân vị 95% (`[2.5%, 97.5%]`) của `Macro-F1 Toxicity` cho từng seed.
   - **QWK (Quadratic Weighted Kappa)** tính trên đủ 4 mức `0..3` (chỉ số bổ trợ đo độ tương hợp có xét thứ tự mức độ độc hại, phạt nặng theo bình phương khoảng cách khi đoán lệch xa nhãn thật).
   - _Nguyên tắc diễn giải bắt buộc trong báo cáo_: Do lớp 3 có số mẫu rất hiếm, `F1-score` riêng của lớp 3 dao động rất mạnh chỉ với vài mẫu đoán sai. Mọi kết luận đánh giá mô hình chỉ được rút ra từ **`Macro-F1` tổng 4 lớp** và **`Confusion matrix` $4 \times 4$**, đồng thời nêu rõ lưu ý này trong báo cáo.
3. **Phân tích Tỷ lệ Dự đoán Nhầm Mức độ Độc hại (FPR) theo Nhóm Cảm xúc**:
   - Lọc lấy tập con các mẫu có nhãn thật **Toxicity = 0** (hoàn toàn không độc hại).
   - Chia tập con này thành 3 nhóm dựa trên nhãn thật **Sentiment**: `Negative (-1)`, `Neutral (0)`, `Positive (1)`.
   - Trên mỗi nhóm cảm xúc $s \in \{-1, 0, 1\}$, tính **3 tỷ lệ nhầm riêng biệt** (không cộng gộp lại):
     $$\text{FPR}_{0 \to 1}(s) = \frac{\text{Số mẫu có } (y_{\text{tox}} = 0, y_{\text{sent}} = s) \text{ bị đoán thành } \hat{y}_{\text{tox}} = 1}{\text{Tổng số mẫu có } (y_{\text{tox}} = 0, y_{\text{sent}} = s)}$$
     $$\text{FPR}_{0 \to 2}(s) = \frac{\text{Số mẫu có } (y_{\text{tox}} = 0, y_{\text{sent}} = s) \text{ bị đoán thành } \hat{y}_{\text{tox}} = 2}{\text{Tổng số mẫu có } (y_{\text{tox}} = 0, y_{\text{sent}} = s)}$$
     $$\text{FPR}_{0 \to 3}(s) = \frac{\text{Số mẫu có } (y_{\text{tox}} = 0, y_{\text{sent}} = s) \text{ bị đoán thành } \hat{y}_{\text{tox}} = 3}{\text{Tổng số mẫu có } (y_{\text{tox}} = 0, y_{\text{sent}} = s)}$$
   - _Lưu ý_: Định nghĩa đa mức này khác bản chất với định nghĩa FPR nhị phân cũ nên không so sánh trực tiếp số học với các báo cáo nhị phân trước đây.
4. **Đo lường Thông lượng (Throughput) và Độ trễ (Latency)**:
   - Đo tổng thời gian chạy thực tế từ lúc nhận chuỗi văn bản thô của `challenge` cho đến khi ra nhãn cuối cùng (bao gồm: làm sạch `clean_text` + tách token + tra `.bin` cho token mới + `forward` qua mạng + áp $T$ và $\beta$).
   - Ghi rõ cấu hình phần cứng (tên CPU, tên GPU, dung lượng RAM/VRAM) và `batch_size` lúc chạy suy luận.
   - Đo và báo cáo tách biệt 2 kịch bản:
     - **Cold start (Khởi động lạnh)**: Bảng từ điển chỉ chứa đúng $V_{\text{full}}$ của `train + valid`; mọi token mới trong `challenge` gặp lần đầu đều phải gọi trực tiếp vào `cc.vi.300.bin` để trích xuất vector rồi mới thêm vào bảng.
     - **Warm start (Khởi động nóng)**: Chạy lại lượt dự đoán ngay sau khi bảng bộ nhớ đệm đã lưu sẵn các vector của token mới từ lượt Cold start, phản ánh tốc độ suy luận khi hệ thống đã đi vào trạng thái ổn định.

### 7.3. Quy tắc Đóng băng Sau Đánh giá

Sau khi chạy xong Phase 7 trên tập `challenge` và xuất báo cáo:

- Toàn bộ pipeline kết thúc.
- Không thực hiện bất kỳ chỉnh sửa nào đối với tiền xử lý, kiến trúc, siêu tham số hay hệ số hậu xử lý rồi chấm lại trên `challenge`, vì bất kỳ vòng lặp chỉnh sửa nào sau khi xem điểm `challenge` đều làm mất tính chất tập mù của dữ liệu kiểm tra cuối cùng.
