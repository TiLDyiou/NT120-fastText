# Báo cáo Quyết định Tuning 3 Vòng (Phase 4)

## 1. Kết quả Vòng 1 — Khảo sát Cơ chế (`seed = 1`)

| Mã | Emb Mode | Dense300 | Task Mode | Loss | F1 Sent | F1 Tox | **Combined F1** | Tham số học được |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `C0` | `freeze_then_open` | `False` | `multi_task` | `weighted_ce` | 0.6322 | 0.5570 | **0.5946** | 7,045,151 |
| `C1a` | `open` | `False` | `multi_task` | `weighted_ce` | 0.6961 | 0.5766 | **0.6363** | 7,045,151 |
| `C1b` | `frozen` | `False` | `multi_task` | `weighted_ce` | 0.5986 | 0.5578 | **0.5782** | 588,551 |
| `C2` | `freeze_then_open` | `True` | `multi_task` | `weighted_ce` | 0.6811 | 0.5469 | **0.6140** | 7,135,451 |
| `C3` | `freeze_then_open` | `False` | `single_task` | `weighted_ce` | 0.6717 | 0.5679 | **0.6198** | 14,089,847 |
| `C4a` | `freeze_then_open` | `False` | `multi_task` | `focal_tox` | 0.6558 | 0.5654 | **0.6106** | 7,045,151 |
| `C4b` | `freeze_then_open` | `False` | `multi_task` | `uncertainty` | 0.6297 | 0.5635 | **0.5966** | 7,045,151 |
| `C_comb` | `open` | `True` | `single_task` | `focal_tox` | 0.6780 | 0.5815 | **0.6298** | 14,270,447 |

- **Chốt khung chuyển sang Vòng 2:** `best_single (C1a) because C_comb=0.6298 < 0.6363` (`emb=open`, `dense300=False`, `task=multi_task`, `loss=weighted_ce`).

## 2. Kết quả Vòng 3 — Xác nhận Đa Seed (`seed ∈ {1, 2, 3}`)

| Hạng | Cấu hình | Sent F1 (Mean) | Tox F1 (Mean) | **Combined F1 (Mean ± Std)** | Tham số học được |
| :---: | :--- | :---: | :---: | :---: | :---: |
| #1 | `R2_17` | 0.6993 | 0.5906 | **0.6449 ± 0.0047** | 7,045,151 |
| #2 | `R2_14` | 0.6931 | 0.5944 | **0.6438 ± 0.0015** | 7,829,791 |
| #3 | `C1a` | 0.6969 | 0.5741 | **0.6355 ± 0.0008** | 7,045,151 |

## 3. Kiểm tra Thống kê & Quy tắc Tie-break (A vs B)

- **Chênh lệch $\mu_A - \mu_B$:** `0.001170` vs $\max(\sigma_A, \sigma_B) = 0.004706$ → Seed Std Tied = `True`
- **Paired Bootstrap 95% CI (1000 lần):** `[-0.003958, 0.006062]` → Chứa 0 = `True`
- *(Tham chiếu A vs C)* $\mu_A - \mu_C = 0.009451$ (max std `0.004706`, tied=`False`), Bootstrap CI `[0.002793, 0.016967]` (chứa 0=`False`).

### Kết luận chọn cấu hình chiến thắng $C^*$: **`R2_17`**
- **Lý do:** Tie-break triggered; A has fewer trainable params.
