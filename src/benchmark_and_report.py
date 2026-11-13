#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Benchmark Mù (Blind Test) & Đánh giá Đa Seed cho Toxicity 4 Mức.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import keras

EMO_NAMES = ["Negative (-1)", "Neutral (0)", "Positive (1)"]
TOX_NAMES = ["T0_NonToxic (0)", "T1_Unpolite (1)", "T2_Insult (2)", "T3_HateThreat (3)"]

EMO_MAP = {"negative": 0, "neg": 0, "neutral": 1, "neu": 1, "positive": 2, "pos": 2,
           "-1": 0, "-1.0": 0, -1: 0, "0": 1, "0.0": 1, 0: 1, "1": 2, "1.0": 2, 1: 2}
TOX_MAP = {
    "0": 0, "0.0": 0, 0: 0, "non-toxic": 0, "non_toxic": 0, "nontoxic": 0, "clean": 0, "t0": 0,
    "1": 1, "1.0": 1, 1: 1, "t1": 1,
    "2": 2, "2.0": 2, 2: 2, "t2": 2,
    "3": 3, "3.0": 3, 3: 3, "t3": 3,
}


def encode(texts, vocab, max_len):
    X = np.zeros((len(texts), max_len), dtype="int32")
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()][:max_len]
        X[i, :len(ids)] = ids
    return X


def unpack(pred):
    if isinstance(pred, dict):
        return pred["emotion"], pred["toxicity"]
    return pred[0], pred[1]


def ece_score(probs, y, bins=10):
    conf, ok = probs.max(1), probs.argmax(1) == y
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def evaluate_single_run(model, cfg, df, dataset_name):
    from preprocess_fasttext import clean

    texts_raw = df["Comment"].astype(str).tolist()
    ye = df["Sentiment"].astype(str).str.strip().str.lower().map(EMO_MAP).values.astype(int)
    yt = df["Toxicity"].astype(str).str.strip().str.lower().map(TOX_MAP).values.astype(int)

    t0 = time.perf_counter()
    texts_clean = [clean(t) for t in texts_raw]
    X = encode(texts_clean, cfg["vocab"], cfg["max_len"])
    pe, pt = unpack(model.predict(X, batch_size=256, verbose=0))
    t1 = time.perf_counter()

    elapsed = t1 - t0
    n = len(df)
    throughput = n / elapsed if elapsed > 0 else 0
    latency_ms = (elapsed / n) * 1000 if n > 0 else 0

    ae = pe.argmax(1)
    at = pt.argmax(1)

    f1_emo = f1_score(ye, ae, average="macro", labels=[0, 1, 2], zero_division=0)
    f1_tox = f1_score(yt, at, average="macro", labels=[0, 1, 2, 3], zero_division=0)
    comb_f1 = (f1_emo + f1_tox) / 2

    # Tỷ lệ báo nhầm Toxic (đoán 1, 2 hoặc 3) trên câu Non-toxic gốc (T0) theo nhóm Cảm xúc
    fp_rates = {}
    for i, en in enumerate(["negative", "neutral", "positive"]):
        m = (yt == 0) & (ye == i)
        fp_rates[en] = float((at[m] != 0).mean()) if m.any() else 0.0

    preds_df = pd.DataFrame({
        "text": texts_raw,
        "text_clean": texts_clean,
        "y_emotion": ye,
        "pred_emotion": ae,
        "y_toxicity": yt,
        "pred_toxicity": at,
    })
    for i, tn in enumerate(["p_t0", "p_t1", "p_t2", "p_t3"]):
        preds_df[tn] = pt[:, i]

    return {
        "dataset": dataset_name,
        "n": n,
        "elapsed_sec": elapsed,
        "throughput": throughput,
        "latency_ms": latency_ms,
        "macro_f1_emo": f1_emo,
        "macro_f1_tox": f1_tox,
        "comb_f1": comb_f1,
        "ece_emo": ece_score(pe, ye),
        "ece_tox": ece_score(pt, yt),
        "fp_rates": fp_rates,
        "report_emo": classification_report(ye, ae, labels=[0, 1, 2], target_names=EMO_NAMES, output_dict=True, zero_division=0),
        "report_tox": classification_report(yt, at, labels=[0, 1, 2, 3], target_names=TOX_NAMES, output_dict=True, zero_division=0),
        "cm_emo": confusion_matrix(ye, ae, labels=[0, 1, 2]).tolist(),
        "cm_tox": confusion_matrix(yt, at, labels=[0, 1, 2, 3]).tolist(),
        "preds_df": preds_df
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["runs/bigru_fasttext_seed1", "runs/bigru_fasttext_seed2", "runs/bigru_fasttext_seed3"],
                    help="Danh sách thư mục runs")
    ap.add_argument("--challenge", default="merged/challenge/challenge.csv")
    ap.add_argument("--subsets", nargs="*", default=[
        "merged/challenge/An/challenge.csv",
        "merged/challenge/Dai/challenge.csv",
        "merged/challenge/Huy/challenge.csv"
    ])
    ap.add_argument("--out", default="reports/blind_benchmark_report.md")
    a = ap.parse_args()

    chal_path = Path(a.challenge)
    if not chal_path.exists():
        alt = list(Path(".").glob("**/challenge.csv"))
        if alt:
            chal_path = alt[0]

    df_chal = pd.read_csv(chal_path, encoding="utf-8-sig")
    df_chal.columns = [c.strip().replace("\ufeff", "") for c in df_chal.columns]

    loaded_runs = []
    for rdir in a.runs:
        rp = Path(rdir)
        m_file = rp / "model.keras"
        v_file = rp / "vocab.json"
        if m_file.exists() and v_file.exists():
            model = keras.saving.load_model(m_file)
            cfg = json.loads(v_file.read_text(encoding="utf-8"))
            loaded_runs.append((rp.name, model, cfg))

    if not loaded_runs:
        raise RuntimeError("Không nạp được thư mục run nào!")

    chal_results = []
    for rname, model, cfg in loaded_runs:
        print(f"--- Đang benchmark {rname} trên Challenge (Toxicity 4 Mức) ---")
        res = evaluate_single_run(model, cfg, df_chal, rname)
        chal_results.append(res)
        out_preds = Path("reports") / f"preds_challenge_{rname}.csv"
        out_preds.parent.mkdir(parents=True, exist_ok=True)
        res["preds_df"].to_csv(out_preds, index=False, encoding="utf-8-sig")

    f1_emos = [r["macro_f1_emo"] for r in chal_results]
    f1_toxs = [r["macro_f1_tox"] for r in chal_results]
    comb_f1s = [r["comb_f1"] for r in chal_results]
    throughputs = [r["throughput"] for r in chal_results]
    latencies = [r["latency_ms"] for r in chal_results]

    mean_emo, std_emo = np.mean(f1_emos), np.std(f1_emos)
    mean_tox, std_tox = np.mean(f1_toxs), np.std(f1_toxs)
    mean_comb, std_comb = np.mean(comb_f1s), np.std(comb_f1s)
    mean_thr = np.mean(throughputs)
    mean_lat = np.mean(latencies)

    primary_name, primary_model, primary_cfg = loaded_runs[0]
    subset_results = []
    for spath in a.subsets:
        p = Path(spath)
        if not p.exists():
            matches = list(Path(".").glob(f"**/{p.parent.name}/{p.name}"))
            if matches:
                p = matches[0]
        if p.exists():
            df_sub = pd.read_csv(p, encoding="utf-8-sig")
            df_sub.columns = [c.strip().replace("\ufeff", "") for c in df_sub.columns]
            sub_res = evaluate_single_run(primary_model, primary_cfg, df_sub, p.parent.name)
            subset_results.append(sub_res)

    lines = [
        "# BÁO CÁO BENCHMARK ĐỘC LẬP (BLIND TEST) - TOXICITY 4 MỨC",
        "## MÔ HÌNH: fastText 300d + Bi-GRU-LSTM-CNN (ĐA NHIỆM)",
        "",
        "> [!IMPORTANT]",
        "> - Phân loại Toxicity ở chế độ 4 mức chi tiết (0: Không độc, 1: Thiếu lịch sự, 2: Xúc phạm, 3: Thù ghét/Đe dọa).",
        "",
        "## 1. Bảng Xếp Hạng Hiệu Năng Đa Seed (Model Leaderboard)",
        "",
        "| Run / Seed | Sentiment Macro F1 | Toxicity (4 Lớp) Macro F1 | **Combined Macro F1** | Thông lượng (mẫu/s) | Độ trễ (ms/câu) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in chal_results:
        lines.append(
            f"| `{r['dataset']}` | {r['macro_f1_emo']:.4f} | {r['macro_f1_tox']:.4f} | **{r['comb_f1']:.4f}** | {r['throughput']:.0f} | {r['latency_ms']:.2f} ms |"
        )

    lines.extend([
        f"| **TRUNG BÌNH (Mean ± Std)** | **{mean_emo:.4f} ± {std_emo:.4f}** | **{mean_tox:.4f} ± {std_tox:.4f}** | **{mean_comb:.4f} ± {std_comb:.4f}** | **{mean_thr:.0f} mẫu/s** | **{mean_lat:.2f} ms** |",
        "",
        "---",
        "",
        f"## 2. Phân Tích Chi Tiết Trên Run Tiêu Biểu ({primary_name})",
        "",
        "### 2.1. Trục 1: Phân Loại Cảm Xúc (Sentiment - 3 Lớp)",
        f"- **Macro F1:** **`{chal_results[0]['macro_f1_emo']:.4f}`** | **Sai số ECE:** `{chal_results[0]['ece_emo']:.4f}`",
        "```text",
        pd.DataFrame(chal_results[0]['report_emo']).T.to_string(),
        "```",
        "",
        "### 2.2. Trục 2: Phân Loại Độ Độc Hại (Toxicity - 4 Lớp)",
        f"- **Macro F1:** **`{chal_results[0]['macro_f1_tox']:.4f}`** | **Sai số ECE:** `{chal_results[0]['ece_tox']:.4f}`",
        "```text",
        pd.DataFrame(chal_results[0]['report_tox']).T.to_string(),
        "```",
        "",
        "### 2.3. Tỷ Lệ Báo Nhầm Toxic (Đoán T1-T3 trên câu T0) Theo Nhóm Cảm Xúc",
        "| Nhóm Cảm Xúc | Tỷ lệ báo nhầm (FPR) |",
        "| :--- | :---: |",
    ])

    for emo, rate in chal_results[0]["fp_rates"].items():
        lines.append(f"| **{emo.upper()}** | **{rate*100:.2f}%** |")

    if subset_results:
        lines.extend([
            "",
            "---",
            "",
            "## 3. Đánh Giá Trên Tập Phân Rã Của Từng Người Gán Nhãn",
            "",
            "| Người gán nhãn | Số mẫu | Sentiment Macro F1 | Toxicity (4 Lớp) Macro F1 | **Combined Macro F1** |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for sr in subset_results:
            lines.append(f"| **{sr['dataset']}** | {sr['n']:,} | {sr['macro_f1_emo']:.4f} | {sr['macro_f1_tox']:.4f} | **{sr['comb_f1']:.4f}** |")

    out_file = Path(a.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[THÀNH CÔNG] Báo cáo Benchmark đã xuất tại: {out_file}")


if __name__ == "__main__":
    main()