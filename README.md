# NT120-fastText (v2)

Dự án này xây dựng một mô hình học sâu đọc bình luận tiếng Việt trên mạng xã hội và gắn cho mỗi bình luận hai nhãn cùng lúc. Nhãn thứ nhất cho biết thái độ của người viết. Nhãn thứ hai cho biết mức độ độc hại của câu chữ.

| Trục | Giá trị | Ý nghĩa |
| :--- | :---: | :--- |
| Sentiment | -1 | Tiêu cực |
| Sentiment | 0 | Trung tính |
| Sentiment | 1 | Tích cực |
| Toxicity | 0 | Không độc hại |
| Toxicity | 1 | Thiếu lịch sự |
| Toxicity | 2 | Xúc phạm cá nhân |
| Toxicity | 3 | Thù ghét hoặc đe dọa |

Mô hình dùng vector từ tiếng Việt **fastText 300 chiều** (`cc.vi.300.bin`) kết hợp mạng nơ-ron song song **Bi-GRU + Conv1D** trên **PyTorch**. Quá trình huấn luyện chạy trên GPU (mất khoảng 2 phút cho mỗi lần huấn luyện mô hình cuối trên RTX 3050), còn lúc suy luận chạy được trên cả GPU và CPU. File trọng số mỗi seed nặng **30,6 MB**.

Đây là **phiên bản làm lại (v2)** theo bản thiết kế `PIPELINE_SPEC.md`, khắc phục tình trạng tụt điểm và nhảy loạn giữa các seed của phiên bản Keras v1 trước đó khi đánh giá Toxicity đủ 4 mức.

## Những gì đã thực hiện

1. Làm sạch tối giản theo đúng 4 bước (chuẩn hóa Unicode NFC, đổi link thành `url`, gộp khoảng trắng, bỏ dòng rỗng) và kiểm tra rò rỉ dữ liệu giữa các tập (`train` còn 29.474 bình luận, `valid` 7.756 bình luận, trùng nhau bằng 0).
2. Tách câu theo đơn vị âm tiết và giữ nguyên cụm ký hiệu/emoticon (không dùng `pyvi`, không hạ chữ thường).
3. Trích xuất ma trận 300 chiều từ file nhị phân `cc.vi.300.bin` (không dùng `<unk>`), đồng thời chuẩn bị cơ chế tra cứu động cho từ mới gặp lúc dự đoán.
4. Dò mô hình qua 3 vòng trên tập `valid` (tổng 42 lần huấn luyện):
   - Vòng 1 (8 lần): so sánh từng cơ chế kiến trúc và chế độ học vector từ.
   - Vòng 2 (25 lần): dò ngẫu nhiên các siêu tham số (`lr`, số vòng, `dropout`, số unit, trọng số lớp).
   - Vòng 3 (9 lần): chạy lại Top-3 cấu hình trên 3 seed độc lập (`1, 2, 3`) và chọn mô hình cuối bằng kiểm tra nhiễu seed kết hợp Paired Bootstrap 1.000 lần.
5. Hiệu chuẩn độ tự tin bằng Temperature Scaling ($T$) và khảo sát hệ số điều chỉnh $\beta$ cho lớp hiếm trên tập `valid`.
6. Huấn luyện lại cấu hình thắng (`R2_17`) trên toàn bộ `train + valid` (**37.230 bình luận**) với 3 seed độc lập (không gộp ensemble) và kiểm tra nạp lại từ đĩa khớp tuyệt đối.
7. Đánh giá mù **đúng một lần duy nhất** trên tập `challenge` (**3.237 bình luận**) và viết báo cáo phân tích lỗi. Toàn bộ quy trình đều có 21 bài tự kiểm tra (unit test) đi kèm.

| Tên mô hình | Dữ liệu huấn luyện | Đặc điểm chính | Số tham số | Vị trí |
| :--- | :--- | :--- | :---: | :--- |
| Mô hình cuối (`seed_1..3`) | `train + valid`, 37.230 bình luận | Cấu hình `R2_17`, đã gắn $T_{\text{sent}}, T_{\text{tox}}$ | 8.030.051 | `runs/final_models/seed_{1,2,3}/` |
| Bản tuning thắng (`R2_17`) | `train`, 29.474 bình luận | Đánh giá trên `valid` ở Vòng 3 & Phase 5 | 7.045.151 | `configs/best_config.json` |

## Thuật toán và lý do lựa chọn

Trong nhánh này, nhóm dùng vector fastText và mạng nơ-ron để làm cầu nối giữa mô hình tuyến tính (TF-IDF + LinearSVC) và mô hình ngôn ngữ lớn (BaMiBERT). Phần dưới giải thích từng khâu hoạt động ra sao và vì sao chọn như vậy. Với mỗi quyết định, README ghi rõ quyết định nào **có số đo trực tiếp** và quyết định nào **dựa trên lập luận thiết kế**.

```text
Bình luận thô
  -> clean_text           chuẩn hóa NFC, đổi link thành url, gộp khoảng trắng
  -> tokenize             tách theo âm tiết và cụm ký hiệu, không <unk>
  -> nn.Embedding(300d)   khởi tạo từ cc.vi.300.bin (tra thêm từ mới lúc dự đoán)
  -> Dropout (0.5)
  -> Song song:
       Bi-GRU (128 unit)             đọc 2 chiều bắt ngữ cảnh dài
       Conv1D (k=2 và k=3, 128)      bắt cụm 2-3 âm tiết cục bộ
  -> Masked Max + Avg Pooling        gom thành vector 1.024 chiều (bỏ qua padding)
  -> Dense(64) -> 2 đầu ra
       Sentiment (3 lớp)   chia T = 1,4695 -> chọn nhãn -1 / 0 / 1
       Toxicity  (4 lớp)   chia T = 1,2857 -> chọn nhãn 0 / 1 / 2 / 3
```

### Đặc điểm dữ liệu chi phối các lựa chọn

Trục Toxicity lệch rất mạnh. Trong tập `train` (29.474 bình luận), lớp 0 có 25.940 câu, lớp 1 có 2.191, lớp 2 có 1.151, còn lớp 3 chỉ có 192 câu (~0,65%). Lớp đông nhất gấp **135 lần** lớp ít nhất. Trục Sentiment lệch nhẹ hơn, khoảng 2,4 lần (`-1`: 6.014, `0`: 14.406, `1`: 9.054).

Độ lệch này dẫn đến hai hệ quả. Thứ nhất, độ chính xác tổng thể không phản ánh đúng chất lượng: một mô hình đoán mọi câu là "không độc hại" vẫn đúng khoảng 88% trên `train` và 84,5% trên `challenge`. Vì vậy mọi bước từ dò tham số đến đánh giá cuối đều lấy **Macro F1** (tính đều F1 của cả 4 lớp) làm thước đo chính. Thứ hai, lớp 3 rất ít nên nếu phạt lỗi quá nặng hoặc chỉnh nhiều ngưỡng trên tập `valid` (chỉ có 54 câu lớp 3) thì điểm rất dễ nhảy loạn hoặc học vẹt.

### 1. Làm sạch tối thiểu và tách âm tiết

`clean_text` chỉ làm đúng 4 việc: đưa Unicode về dạng dựng sẵn `NFC`, thay đường link (`http...`, `www...`) bằng chữ `url`, gộp khoảng trắng thừa và bỏ dòng trống.

Khác với bản Keras v1 cũ, bộ làm sạch v2 **cố ý không làm** những việc sau:

- Không hạ chữ thường (`lower()`). Trong `cc.vi.300.bin` có phân biệt chữ hoa và chữ thường; câu viết hoa kiểu `"ĐẸPPPP"` mang sắc thái khác chữ thường.
- Không dùng `pyvi` tách từ ghép. Bộ `cc.vi.300.bin` được huấn luyện trên văn bản tiếng Việt tách theo khoảng trắng (đơn vị âm tiết). Dùng `pyvi` rất dễ băm vụn emoticon kiểu `=)))` thành `= ) ) )` và sai với từ lóng mạng xã hội. Việc ghép các từ 2–3 âm tiết như `"dịch vụ"` hay `"đồ ngu"` được giao cho tầng `Conv1D` bên dưới.
- Không chuẩn hóa teencode, không xóa ký tự lặp (`nguuuu`), không xóa dấu chấm chèn giữa chữ (`đ.m`, `l.ồ.n`).

Bước chuẩn hóa `NFC` không phải lý thuyết suông: khi quét tập `valid`, có **56 bình luận** dùng bảng mã tổ hợp (NFD). Nếu không chuẩn hóa về `NFC`, cùng một chữ có dấu sẽ bị fastText cắt ký tự con thành hai vector khác nhau.

Khâu tách token chia câu theo khoảng trắng, bóc cụm dấu câu ở đầu và cuối mỗi chữ nhưng giữ nguyên dấu xen giữa (`đ.m`, `ngục.mày`). Cụm toàn ký hiệu (`=)))`, `:))`, `???`) được giữ nguyên làm 1 token. Có một chi tiết nhỏ thực tế: ký hiệu `<3` chứa chữ số `3` nên quy tắc chung tách nó thành `['<', '3']`. Khi kiểm tra thẳng file `cc.vi.300.bin` (vốn dùng độ dài ký tự con tối thiểu `minn=5`), `<3` nguyên cụm bị trả về vector toàn số 0, trong khi tách thành `'<'` và `'3'` lại cho hai vector khác 0 và tầng `Conv1D(k=2)` vẫn ghép lại được. Vì vậy code giữ nguyên quy tắc chung, không chế ngoại lệ riêng cho `<3`.

*Điểm cần nói thẳng:* bản v2 thay đổi cả tiền xử lý lẫn kiến trúc cùng lúc so với bản v1, nên chưa chạy thực nghiệm tách riêng để đo xem riêng việc bỏ `pyvi` và giữ chữ hoa đóng góp bao nhiêu phần trăm điểm.

### 2. Dùng file `.bin` và mở rộng từ điển lúc dự đoán

Các mô hình vector từ cổ điển dùng file `.vec` thường gặp bế tắc khi gặp từ lạ (OOV): bất kỳ từ viết sai chính tả hay từ lóng mới nào xuất hiện ở tập test đều bị gán chung vào một ô `<unk>`.

Pipeline v2 bắt buộc dùng file nhị phân `cc.vi.300.bin` và **bỏ hoàn toàn token `<unk>`**. File `.bin` lưu toàn bộ bảng trọng số các đoạn ký tự con (subword n-grams). Nhờ vậy, khi gặp một từ chưa từng có trong tập `train` (chẳng hạn biến thể `"xamlone"` hay `"ngục.mày"`), fastText vẫn cộng các mảnh ký tự con để tạo ra một vector 300 chiều có nghĩa.

Vì `cc.vi.300.bin` khi nạp vào RAM tốn khoảng 7 GB, pipeline tách bước rút trích ma trận thành một bước chạy trước khi huấn luyện:

- Lúc huấn luyện trên `train`, bảng `nn.Embedding` chỉ chứa đúng 21.521 từ của `train` cộng thêm 1 dòng `<pad>` bằng 0 ở vị trí 0.
- 3.283 từ mới chỉ xuất hiện trong tập `valid` được rút sẵn vector gốc ra một file riêng (`valid_oov_vectors.npy`), **tuyệt đối không nằm trong optimizer lúc học** để không nhận gradient hay weight decay.
- Lúc dự đoán trên `valid` hay `challenge`, `DynamicPredictor` gặp từ nào mới thì tra từ file pre-cache hoặc trực tiếp từ `cc.vi.300.bin` rồi nối thêm dòng vào cuối bảng tra cứu (không cập nhật trọng số). Trên tập `challenge`, cơ chế này đã tự động cấp vector cho **1.079 token mới**.

### 3. Kiến trúc song song `Bi-GRU ‖ Conv1D(2, 3)` và che padding chặt

Bản Keras v1 cũ xếp 3 mạng nối tiếp `Bi-GRU -> LSTM -> Conv1D`, vừa nặng hơn vừa không che kỹ phần đệm số 0 (`padding`) ở cuối câu ngắn.

Bản v2 chuyển sang **kiến trúc song song** đặt ngay sau tầng Embedding:

- **Nhánh `Bi-GRU` (128 unit mỗi chiều):** đọc câu từ trái sang phải và phải sang trái để hiểu các mẫu phụ thuộc xa như câu phủ định hoặc chuyển ý. Nhờ dùng `pack_padded_sequence` theo đúng độ dài thật `actual_len`, chiều đọc ngược bắt đầu chính xác từ từ cuối cùng của câu, không bị đọc nhầm các số `0` đệm ở cuối.
- **Hai nhánh `Conv1D` (kernel = 2 và kernel = 3, mỗi nhánh 128 bộ lọc):** trượt cửa sổ 2 và 3 token để bắt trực tiếp các từ ghép tiếng Việt (2–3 âm tiết) và cụm từ chửi thề/cảm thán cục bộ. Trước và sau khi qua `Conv1D`, mọi vị trí padding đều bị nhân với mặt nạ `mask = 0` (với `k=2`, chuỗi được đệm bất đối xứng 1 cột bên phải) để giá trị `bias` của lớp tích chập không bị tràn sang vùng trống.
- **Gộp đặc trưng:** nối đầu ra của 3 nhánh thành tensor 4H = 512 chiều cho mỗi vị trí, sau đó lấy **Masked Max Pooling** (gán `-1e9` vào vùng padding trước khi lấy max) và **Masked Avg Pooling** (chia đúng cho `actual_len`) rồi nối thành vector 1.024 chiều đưa vào tầng ẩn 64 nút và 2 đầu ra logits.

Cách đặt song song rút ngắn đường truyền gradient từ hàm loss về thẳng tầng Embedding thay vì phải đi xuyên qua 3 tầng mạng như bản cũ. Phần thân mạng chỉ có **588.551 tham số** (phần lớn kích thước 30 MB của file mô hình nằm ở bảng Embedding 24.805 × 300).

### 4. Học đa nhiệm và kết quả khảo sát cơ chế ở Vòng 1

Đề tài có hai trục liên quan nhau (câu độc hại thường mang cảm xúc tiêu cực; trong 246 câu `T3` ở `train + valid`, có 239 câu mang nhãn Tiêu cực và chỉ 7 câu Trung tính hoặc Tích cực). Vì vậy mô hình mặc định dùng một thân mạng chung rẽ ra hai đầu ra (`multi_task`).

Trong Vòng 1 trên tập `valid` (`seed = 1`), nhóm thử đổi từng yếu tố một so với cấu hình gốc `C0` (`freeze_then_open` đóng băng 2 vòng đầu, không Dense300, đa nhiệm, loss `weighted_ce`):

| Mã | Thay đổi duy nhất so với `C0` | Sentiment F1 | Toxicity F1 | **Combined F1** | Tham số học được |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `C0` | Cấu hình gốc (`freeze_then_open`, F=2) | 0,6322 | 0,5570 | **0,5946** | 7.045.151 |
| **`C1a`** | **Mở khóa Embedding ngay từ vòng 1 (`open`)** | **0,6961** | **0,5766** | **0,6363** | 7.045.151 |
| `C1b` | Đóng băng Embedding suốt quá trình học (`frozen`) | 0,5986 | 0,5578 | 0,5782 | 588.551 |
| `C2` | Thêm tầng chiếu `Dense 300` sau Embedding | 0,6811 | 0,5469 | 0,6140 | 7.135.451 |
| `C3` | Tách 2 mạng đơn nhiệm độc lập (`single_task`) | 0,6717 | 0,5679 | 0,6198 | 14.089.847 |
| `C4a` | Đổi loss Toxicity sang `Focal Loss` ($\gamma=2$) | 0,6558 | 0,5654 | 0,6106 | 7.045.151 |
| `C4b` | Đổi loss sang `Uncertainty Weighting` | 0,6297 | 0,5635 | 0,5966 | 7.045.151 |
| `C_comb` | Ghép các nhánh thắng (`open` + `Dense300` + `single_task` + `focal`) | 0,6780 | 0,5815 | 0,6298 | 14.270.447 |

Số đo Vòng 1 cho thấy ba điểm rất rõ:

- **Mở Embedding ngay (`open`) thắng lớn:** Với ngân sách huấn luyện ngắn ($E = 5$ vòng) và learning rate giảm theo hình cosin, việc đóng băng 2 vòng đầu ở `C0` khiến các vector từ lóng/teencode không kịp điều chỉnh ở 3 vòng cuối (Combined F1 chỉ đạt `0,5946`). Mở khóa ngay từ vòng 1 (`C1a`) giúp vector nhanh chóng thích nghi với dữ liệu mạng xã hội, đẩy Sentiment F1 lên `0,6961` và tổng điểm lên `0,6363`. Đóng băng hoàn toàn (`C1b`) tụt sâu xuống `0,5782`, khẳng định fine-tuning vector là bắt buộc.
- **Đa nhiệm tốt và nhẹ hơn đơn nhiệm:** Tách 2 mạng riêng (`C3`) tốn gấp đôi số tham số (14,09 triệu) nhưng chỉ đạt `0,6198`, thua `C1a`.
- **Tổ hợp `C_comb` thua cấu hình đơn lẻ `C1a` (`0,6298 < 0,6363`):** Theo đúng quy tắc chốt khung viết sẵn, vòng 2 khóa khung của **`C1a`** (`open`, không Dense300, đa nhiệm, `weighted_ce`).

### 5. Trọng số lớp làm dịu cho Toxicity

Đây là thay đổi tạo ra khác biệt lớn nhất về độ ổn định giữa bản v1 và bản v2.

Ở bản Keras v1, dùng trọng số nghịch đảo tần suất đầy đủ (`balanced`) khiến mỗi câu lớp 3 bị phạt nặng hơn lớp 0 tới hơn 130 lần. Vì lớp 3 chỉ có 192 câu trong tập `train`, vài câu khó trong một batch ngẫu nhiên có thể tạo gradient khổng lồ làm lệch toàn bộ mô hình (đó là lý do ở bản v1, Seed 2 bị tụt điểm Toxicity từ 0,6216 xuống 0,4993).

Bản v2 dùng công thức làm dịu có giới hạn trần cho lớp Toxicity $k \in \{0, 1, 2, 3\}$:

$$
w_k = \min\left( \left( \frac{N}{4 \cdot n_k} \right)^\alpha, \, C_{\max} \right)
$$

Với cấu hình thắng ($\alpha = 0,5$, trần trọng số $C_{\max} = 10,0$, tức `clip_max = 10.0` trong code), trọng số 4 lớp trên tập `train + valid` lần lượt là `0,53`, `1,84`, `2,55` và `6,15` (lớp 3 chỉ nặng gấp khoảng **11,5 lần** lớp 0 thay vì hơn 130 lần). Nhờ đó độ lệch chuẩn giữa 3 seed trên tập `challenge` giảm tới **~14 lần** (từ `0,0531` xuống `0,0037`). Trục Sentiment phân phối đều hơn nên giữ trọng số bằng 1.

### 6. Lịch học cố định và không dừng sớm

Thay vì dùng EarlyStopping canh điểm `valid` như bản v1, bản v2 chạy hết đúng $E = 5$ epoch với `AdamW` (`batch_size = 64`), cắt gradient ở chuẩn 1,0 và giảm learning rate mượt từ `1e-3` về `0` bằng `CosineAnnealingLR`.

Lý do thiết kế: đến Phase 6, mô hình cuối được huấn luyện trên toàn bộ `train + valid`, lúc đó **không còn tập `valid`** để canh dừng sớm nữa. Việc xem số vòng $E$ là một siêu tham số cố định giúp lần huấn luyện cuối đi theo đúng quỹ đạo tối ưu như lúc dò tham số.

### 7. Dò siêu tham số 3 vòng và luật chọn đơn giản hơn

Sau khi chốt khung `C1a` ở Vòng 1, Vòng 2 lấy mẫu ngẫu nhiên 25 cấu hình không trùng lặp (`seed = 1`). Ở Vòng 3, 3 cấu hình dẫn đầu được chạy lại đủ 3 seed (`1, 2, 3`):

| Hạng | Cấu hình | Sent F1 (Mean) | Tox F1 (Mean) | **Combined F1 (Mean ± Std)** | Tham số học được |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | **`R2_17`** | 0,6993 | 0,5906 | **0,6449 ± 0,0047** | **7.045.151** |
| #2 | `R2_14` | 0,6931 | 0,5944 | 0,6438 ± 0,0015 | 7.829.791 |
| #3 | `C1a` | 0,6969 | 0,5741 | 0,6355 ± 0,0008 | 7.045.151 |

Chênh lệch trung bình giữa `R2_17` và `R2_14` chỉ là `0,0012` — nhỏ hơn độ lệch chuẩn giữa các seed (`0,0047`) và khoảng tin cậy Paired Bootstrap 95% trên `valid` (`[-0,0040; +0,0061]`) chứa giá trị `0`. Hai cấu hình được coi là ngang nhau về mặt thống kê, nên quy tắc tie-break chọn mô hình ít tham số hơn là **`R2_17`** (`H = 128`, `E = 5`, `lr = 0,001`, `dropout = 0,5`, `weight_decay = 0,0`, `alpha = 0,5`, `clip_max = 10,0`, `w_tox = 1,0`).

### 8. Hiệu chuẩn xác suất ($T$) và kiểm tra $\beta$

Mạng nơ-ron học sâu hay bị "quá tự tin" (độ tự tin dự đoán 95% nhưng thực tế chỉ đúng 85%) hoặc bị lệch xác suất do trọng số lớp. Chỉ số **ECE** đo độ lệch giữa xác suất dự đoán và tỉ lệ đúng thực tế (càng gần 0 càng tốt).

Phase 5 chia vector logit đầu ra cho một hằng số dương $T$ (Temperature Scaling) tối ưu trên `valid` để giảm sai số xác suất NLL và ECE:

- Trên `valid` (3 seed `R2_17`): $\bar{T}_{\text{sent}} = \mathbf{1,4695}$ giúp ECE Sentiment giảm hơn **3 lần** (từ `0,0781 ± 0,0030` xuống **`0,0227 ± 0,0030`**); $\bar{T}_{\text{tox}} = \mathbf{1,2857}$ giúp ECE Toxicity giảm **2 lần** (từ `0,0222 ± 0,0024` xuống **`0,0111 ± 0,0015`**).
- Vì chia cho số dương $T$ không làm đổi thứ tự lớn nhỏ giữa các lớp, **Macro F1 trước và sau bước này giữ nguyên tuyệt đối**.
- Tiếp theo, pipeline thử nhân xác suất Toxicity với hệ số bù tần suất $\tau_k(\beta) = (f_0 / f_k)^\beta$ với $\beta \in \{0, 0,25, 0,5, 0,75, 1\}$. Kết quả trên `valid`: $\beta = 0,0$ đạt **`0,5906`**, còn tăng $\beta$ lên `0,25..1,0` làm điểm tụt liên tục (`0,5665 → 0,5352 → 0,4962 → 0,4649`). Lý do là trọng số lớp $\alpha = 0,5$ trong lúc huấn luyện đã bù cho lớp hiếm đủ rồi; nhân thêm $\beta > 0$ lúc suy luận khiến mô hình đoán vống các lớp hiếm. Vì vậy hệ thống chốt $\beta_{\text{final}} = 0,0$.

### 9. Huấn luyện lại trên `train + valid`

Sau khi chốt xong toàn bộ siêu tham số và $T$, mô hình `R2_17` được huấn luyện lại từ đầu trên toàn bộ 37.230 bình luận (`train + valid`) trong đúng 5 epoch cho 3 seed (`1, 2, 3`). Bước này giúp mô hình học thêm được 54 bình luận lớp 3 từ tập `valid` (nâng tổng lớp 3 lên 246 câu). Ba seed được giữ độc lập để báo cáo độ ổn định `Mean ± Std` trên tập mù `challenge`.

## Kết quả đạt được

Chỉ số chính của đề tài là **Macro F1** (tính trung bình đều F1 từng lớp nên lớp hiếm có trọng lượng ngang lớp phổ biến).

### Trên tập `challenge` (3.237 bình luận — chấm đúng 1 lần duy nhất)

| Phiên bản | Sentiment Macro F1 | **Toxicity Macro F1 (4 lớp)** | **Combined Macro F1** | ECE Sent (Seed 1) | ECE Tox (Seed 1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bản cũ v1 (Keras, 4 lớp) | 0,6423 ± 0,0079 | 0,5731 ± 0,0531 | 0,6077 ± 0,0250 | 0,2440 | 0,0776 |
| **Bản mới v2 (PyTorch, 4 lớp)** | **0,6439 ± 0,0067** | **0,6593 ± 0,0037** | **0,6516 ± 0,0038** | **0,0782** | **0,0203** |

Chi tiết từng seed của bản mới v2:

| Seed | Sentiment F1 | Toxicity F1 (4 lớp) | Combined F1 | QWK (Toxicity) | CI 95% Tox F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `seed_1` | 0,6431 | **0,6635** | 0,6533 | 0,6731 | [0,6159; 0,7092] |
| `seed_2` | 0,6376 | **0,6568** | 0,6472 | 0,6769 | [0,6091; 0,7017] |
| `seed_3` | 0,6510 | **0,6575** | 0,6543 | 0,6765 | [0,6141; 0,7022] |
| **Mean ± Std** | **0,6439 ± 0,0067** | **0,6593 ± 0,0037** | **0,6516 ± 0,0038** | **0,6755 ± 0,0021** | — |

Một số con số chi tiết đáng chú ý trên tập `challenge`:

- **Toxicity Macro F1 (4 lớp) tăng +0,0862 (+8,62 điểm %)** so với bản Keras v1 và độ lệch chuẩn giảm **~14 lần**. Điểm tương hợp bậc 4 mức (**QWK**) đạt **0,6755 ± 0,0021**.
- Độ chính xác (Accuracy) trung bình 3 seed: **65,82%** cho Sentiment và **89,98%** cho Toxicity.
- F1 từng lớp Sentiment: `Negative (-1)`: `0,5241` (1.073 câu), `Neutral (0)`: `0,6917` (1.156 câu), `Positive (1)`: `0,7158` (1.008 câu).
- F1 từng lớp Toxicity: `T0`: `0,9522` (2.736 câu), `T1`: `0,5665` (314 câu), `T2`: `0,6298` (158 câu), `T3`: `0,4886` (29 câu). Do lớp `T3` chỉ có 29 câu trong `challenge`, điểm riêng lớp 3 rất nhạy chỉ với vài câu đoán lệch nên mọi đánh giá đều dựa trên Macro F1 tổng 4 lớp và ma trận nhầm lẫn.
- **Tỷ lệ báo nhầm độc hại trên bình luận `T0` (2.736 câu sạch) theo từng nhóm cảm xúc:**

| Nhóm cảm xúc (`y_tox = 0`) | Số câu | Đoán nhầm thành `T1` | Đoán nhầm thành `T2` | Đoán nhầm thành `T3` |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (-1)** | 651 | 4,71% ± 0,90% | 1,43% ± 0,23% | 0,67% ± 0,18% |
| **Neutral (0)** | 1.128 | 1,80% ± 0,45% | 0,56% ± 0,14% | 0,21% ± 0,10% |
| **Positive (1)** | 957 | 1,57% ± 0,31% | 0,07% ± 0,06% | 0,03% ± 0,06% |

*(Định nghĩa FPR tách 3 mức này khác bản nhị phân cũ nên không so sánh số học trực tiếp với các báo cáo nhị phân trước đây.)*

- **Tốc độ suy luận (RTX 3050 Laptop 6GB, `batch_size = 256`):**
  - **Cold start** (gồm làm sạch + tách token + tra `cc.vi.300.bin` cho 1.079 token OOV mới + forward + hậu xử lý): **2.621 ± 774 bình luận/giây** (~0,41 ms/câu).
  - **Warm start** (khi từ mới đã nằm trong cache): **9.537 ± 565 bình luận/giây** (~0,105 ms/câu).

### Trên tập `valid` (lúc dò siêu tham số, train-only)

Trên 7.756 bình luận `valid`, mô hình thắng `R2_17` đạt Sentiment Macro F1 **0,6993** và Toxicity Macro F1 **0,5906** (Combined **0,6449 ± 0,0047**).

Có một điểm ngược nhau khá thú vị: khi sang tập `challenge`, Toxicity tăng từ `0,5906` lên `0,6593`, nhưng Sentiment lại giảm từ `0,6993` xuống `0,6439`. Nguyên nhân đã được ghi nhận trong `LABELING_GUIDELINES.md`: tập `challenge` đã được duyệt lại toàn bộ theo quy tắc mới (mỉa mai/châm chọc đều đưa về `-1`, nâng tỉ lệ `Negative` lên 33,1%), trong khi `train` (20,4% Negative) và `valid` (21,1% Negative) chưa được duyệt lại theo quy tắc này.

### Hạn chế đã biết

- **Bỏ sót câu tiêu cực / mỉa mai ngược:** Trên `challenge` (`seed_1`), mô hình nhận đúng 38,4% câu Negative (412/1.073 câu, nhầm 477 câu sang Neutral), nhưng một khi đã đoán Negative thì đúng tới 80,6%. Các câu kiểu *"Cố mỉm cười nhưng lòng chẳng được vui :)))"* vẫn bị đoán nhầm thành Positive ($P_{\text{pos}} = 0,955$) do vector tĩnh của `"mỉm cười"`, `"vui"`, `":)))"` lấn át.
- **Nhạy cảm với câu VIẾT HOA TOÀN BỘ + gõ sai:** Ví dụ `"MÁY CON ĐIÊN NẦY SAO KHÔNG AI NHỐT LẠI ĐỂ LOẠN HẾT"` bị đoán thành `(Neutral, Non-toxic)` do giữ nguyên chữ hoa khiến cụm từ lệch khỏi phân phối chữ thường phổ biến.
- **Bẫy từ khóa làm báo nhầm Toxic:** Câu *"Chính sách ngu dân ngày càng hiệu quả"* bị đoán thành `T2` ($P_{T2} = 0,994$) vì dính chữ `"ngu"`; câu đùa *"tao méc mấy a dlv tụi bây phản động nè"* bị đoán thành `T3` ($P_{T3} = 0,948$) do chữ `"dlv"`.
- **Bỏ sót công kích ngầm không có từ tục:** Các câu miệt thị gián tiếp như *"Toàn bọn đầu trộm , đuôi cướp !"* hay kỳ thị vùng miền ngầm *"Tương lai thanh niên Bến Tre đổi qua nghề ' Trộm ' hết."* đều bị đoán `T0` ($P_{T0} > 0,99$) vì không chứa từ khóa chửi thề quen thuộc.

Chi tiết các ca lỗi trích trực tiếp từ dữ liệu xem thêm trong [reports/error_analysis.md](reports/error_analysis.md).

## Repo đang có gì

```text
.
├── src_v2/
│   ├── config.py              Đường dẫn, hằng số và lưới siêu tham số
│   ├── check_env.py           Kiểm tra PyTorch CUDA, fasttext và file .bin
│   ├── data_prep.py           Phase 1: làm sạch 4 bước, chống leak, tách token
│   ├── build_embeddings.py    Phase 2: dựng từ điển V_train/V_trainV1/V_full và trích .npy
│   ├── model.py               Phase 3: ParallelNet (Bi-GRU || Conv1D), class weight, 3 loss
│   ├── predict.py             Phase 3: DynamicPredictor + NT120Predictor (API dự đoán)
│   ├── metrics.py             Phase 3: Macro-F1, ECE, QWK, bootstrap CI, FPR đa mức
│   ├── train.py               Phase 3: vòng huấn luyện GPU, AdamW + CosineAnnealingLR
│   ├── tuning.py              Phase 4: dò 3 vòng + hỗ trợ resume + tie-break
│   ├── calibrate.py           Phase 5: Temperature Scaling (float64) + β + kiểm tra V1/V2
│   ├── final_train.py         Phase 6: huấn luyện lại trên train+valid (3 seed)
│   ├── evaluate_challenge.py  Phase 7: đánh giá mù challenge + đo Cold/Warm start
│   └── utils/                 Cố định seed và ghi log JSONL
├── tests/                     21 unit test kiểm chứng toàn bộ 7 Phase
├── data/
│   ├── train/train.csv        29.474 bình luận
│   ├── valid/valid.csv        7.756 bình luận
│   └── challenge/challenge.csv  3.237 bình luận
├── configs/
│   └── best_config.json       Siêu tham số chiến thắng R2_17
├── artifacts/
│   └── postproc.json          Nhiệt độ T và tham số β đã chốt ở Phase 5
├── runs/
│   ├── tuning_log.jsonl       Nhật ký đầy đủ 42 lần chạy dò tham số
│   ├── tuning_cache/          Logits lưu sẵn của Top-3 × 3 seed
│   └── final_models/          3 thư mục seed_1, seed_2, seed_3 (mỗi seed ~31 MB)
├── reports/
│   ├── data_stats.md          Thống kê dữ liệu sau Phase 1
│   ├── ablation_table.csv     Bảng tổng hợp kết quả cả 42 lần chạy
│   ├── tuning_decision.md     Quyết định Vòng 1, Vòng 3 và kết quả tie-break
│   ├── calibration.md         Kết quả ECE trước/sau T và đường cong β
│   ├── challenge_report.md    Báo cáo chính thức trên tập mù challenge
│   ├── challenge_metrics.json Số liệu JSON chi tiết trên challenge
│   ├── error_analysis.md      Phân tích lỗi định tính từ các ca thực tế
│   └── preds_challenge_seed{1,2,3}.csv  Dự đoán chi tiết từng câu của 3 seed
├── PIPELINE_SPEC.md           Bản đặc tả kỹ thuật đầy đủ của nhánh v2
├── requirements.txt           Danh sách thư viện cần cài
└── README.md                  Giải thích repo này
```

Mọi file CSV đầu vào dùng mã hóa UTF-8 và có 3 cột `Comment`, `Sentiment`, `Toxicity`. File nhị phân `embeddings/cc.vi.300.bin` (6,74 GB) không đưa lên Git do vượt giới hạn dung lượng của GitHub.

## Hướng dẫn dùng mô hình

### Cài đặt

Khuyên dùng **Python 3.10 – 3.12** (trên Windows, `fasttext-wheel==0.9.2` chưa có bản build sẵn cho Python 3.13).

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate

# Cài PyTorch (chọn bản CUDA nếu có GPU Nvidia, hoặc bản thường nếu chỉ chạy CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Nếu muốn chạy suy luận đầy đủ cho cả từ mới chưa từng gặp trong lúc huấn luyện (hoặc chạy lại Phase 2 / Phase 7), tải `cc.vi.300.bin.gz` (4,2 GB) từ Meta fastText rồi giải nén vào `embeddings/cc.vi.300.bin`:

```text
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz
```

### Dự đoán cho một vài câu

Chạy đoạn mã sau từ thư mục gốc của repo:

```python
from src_v2.predict import NT120Predictor

# Cách chuẩn (có cc.vi.300.bin để tra subword cho từ mới):
model = NT120Predictor.from_dir(
    "runs/final_models/seed_1",
    fasttext_bin="embeddings/cc.vi.300.bin",
)

# Nếu chỉ muốn chạy thử nhanh trên máy chưa tải file 6.7 GB:
# model = NT120Predictor.from_dir("runs/final_models/seed_1", allow_zero_oov=True)

texts = [
    "Sản phẩm dùng rất tốt, giao hàng nhanh",
    "Đồ ngu, cút đi",
    "Hôm nay trời mưa",
]
print(model.predict(texts))
# Cột 0 là Sentiment (-1/0/1), cột 1 là Toxicity (0/1/2/3)
```

Muốn xem xác suất mô hình dự đoán sau hiệu chuẩn Temperature Scaling, gọi `predict_proba`:

```python
sent_proba, tox_proba = model.predict_proba(texts)
# sent_proba: shape (N, 3) theo thứ tự nhãn -1, 0, 1
# tox_proba : shape (N, 4) theo thứ tự nhãn 0, 1, 2, 3
```

### Dự đoán cho cả file CSV

```python
import pandas as pd
from src_v2.predict import NT120Predictor

model = NT120Predictor.from_dir(
    "runs/final_models/seed_1",
    fasttext_bin="embeddings/cc.vi.300.bin",
)

df = pd.read_csv("input.csv", encoding="utf-8-sig")
pred = model.predict(df["Comment"].astype(str).tolist())
df["pred_sentiment"] = pred[:, 0]
df["pred_toxicity"] = pred[:, 1]
df.to_csv("output.csv", index=False, encoding="utf-8-sig")
```

### Kiểm thử và chạy lại pipeline

Chạy toàn bộ 21 bài tự kiểm tra (tự động kiểm tra lại logic xử lý chữ, padding-invariance, đóng mở gradient, chọn tham số, hiệu chuẩn và nạp mô hình):

```bash
pytest tests -v
```

Nếu muốn chạy lại từ đầu các bước tiền xử lý, dò tham số và huấn luyện lại trên `train + valid` (cần có `embeddings/cc.vi.300.bin`):

```bash
python -m src_v2.check_env --load-bin
python -m src_v2.data_prep
python -m src_v2.build_embeddings
python -m src_v2.tuning --n-random 25
python -m src_v2.calibrate
python -m src_v2.final_train
```

- Bước `tuning` hỗ trợ tự động chạy tiếp (`resume`) từ `runs/tuning_log.jsonl` nếu bị ngắt giữa chừng.
- Tập `challenge` đã được chấm đúng 1 lần duy nhất và đóng băng kết quả tại [reports/challenge_report.md](reports/challenge_report.md).
