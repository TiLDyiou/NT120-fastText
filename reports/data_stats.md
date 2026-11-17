# Thống kê Dữ liệu sau Phase 1 (`data_prep.py`)

> Ghi chú: Để đảm bảo `challenge` là tập mù tuyệt đối, báo cáo này không thống kê phân phối nhãn hay từ vựng của `challenge` (chỉ kiểm tra số dòng trước/sau lọc rỗng).

## 1. Tổng hợp Khử trùng lặp & Chống rò rỉ

| Tập dữ liệu | Dòng gốc | Loại rỗng | Trùng nội bộ | Trùng chéo train∩valid | Còn lại cuối |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train` | 29,474 | 0 | 0 | 0 | **29,474** |
| `valid` | 7,756 | 0 | 0 | 0 | **7,756** |
| `challenge` | 3,237 | 0 | 0 (giữ nguyên) | 0 (giữ nguyên) | **3,237** |

## 2. Phân phối Nhãn (`train` & `valid`)

### Tập `train` ($N = 29,474$)
- **Sentiment (`-1 / 0 / 1`):** `{-1: 6014, 0: 14406, 1: 9054}`
- **Toxicity (`0 / 1 / 2 / 3`):** `{0: 25940, 1: 2191, 2: 1151, 3: 192}`
- **Độ dài token:** min=1, median=10.0, **p95=53**, p99=98.0, max=579

### Tập `valid` ($N = 7,756$)
- **Sentiment (`-1 / 0 / 1`):** `{-1: 1638, 0: 3717, 1: 2401}`
- **Toxicity (`0 / 1 / 2 / 3`):** `{0: 6866, 1: 554, 2: 282, 3: 54}`
- **Độ dài token:** min=1, median=11.0, **p95=57**, p99=104.4, max=433

