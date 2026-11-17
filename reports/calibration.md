# Báo cáo Hiệu chuẩn & Điều chỉnh Hậu xử lý (Phase 5)

Áp dụng trên 3 seed của cấu hình chiến thắng **`R2_17`**.

## 1. Temperature Scaling & Chỉ số ECE (M = 15 bins)

- $\bar{T}_{\text{sent}}$ = **1.4695** (từng seed: `[1.4633, 1.4417, 1.5036]`)
- $\bar{T}_{\text{tox}}$ = **1.2857** (từng seed: `[1.2599, 1.2813, 1.3159]`)

| Nhiệm vụ | ECE Trước $T$ (Mean ± Std) | ECE Sau $\bar{T}$ (Mean ± Std) | NLL Trước → Sau |
| :--- | :---: | :---: | :---: |
| **Sentiment (3 lớp)** | 0.0781 ± 0.0030 | **0.0227 ± 0.0030** | 0.7188 → 0.6773 |
| **Toxicity (4 lớp)** | 0.0222 ± 0.0024 | **0.0111 ± 0.0015** | 0.2985 → 0.2858 |

- Đã kiểm chứng `Macro-F1` trước và sau khi chia cho $\bar{T}$ giữ nguyên tuyệt đối (chênh lệch `< 1e-12`).

## 2. Đường cong Khảo sát $\beta$ trên `valid` (Toxicity 4 lớp)

| $\beta$ | Macro-F1 Toxicity (Mean ± Std 3 seed) |
| :---: | :---: |
| `0.00` | 0.5906 ± 0.0080 *(tối ưu $\beta^*$)* |
| `0.25` | 0.5665 ± 0.0086 |
| `0.50` | 0.5352 ± 0.0079 |
| `0.75` | 0.4962 ± 0.0063 |
| `1.00` | 0.4649 ± 0.0103 |

## 3. Kiểm tra Chuyển giao `V1 / V2`

- $\beta^* = 0.0$ nên không cần bước kiểm tra `V1/V2`; chốt $\beta_{\text{final}} = 0.0$.
