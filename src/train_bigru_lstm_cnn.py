#!/usr/bin/env python3
"""
Train Bi-GRU-LSTM-CNN + fastText 300d (nhánh 2 của đề tài NT120), 2 trục cùng lúc:
    - emotion  : Negative / Neutral / Positive   (3 lớp)
    - toxicity : Non-toxic / Toxic               (2 lớp)
Một bộ "thân" dùng chung, 2 đầu ra (multi-task).

Đầu vào:
  Cách 1: File CSV gộp đã qua prepare_dataset.py (--data data_ft.csv) có các cột:
          text_clean | emotion | toxicity | split
  Cách 2: 2 file riêng biệt qua --train-data và --val-data

Ví dụ:
    python train_bigru_lstm_cnn.py --data data_ft.csv --embeddings cc.vi.300.vec.gz --seed 1
    python train_bigru_lstm_cnn.py --train-data train.csv --val-data valid.csv --embeddings cc.vi.300.vec.gz
    python train_bigru_lstm_cnn.py --predict runs/bigru_seed1 --text "dịch vụ tệ quá đồ lừa đảo"
"""
import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

import keras
from keras import layers

EMO_NAMES = ["negative", "neutral", "positive"]
TOX_NAMES = ["non-toxic", "toxic"]

# Hỗ trợ cả tên chữ lẫn mã số (-1/0/1 hoặc 0/1/2)
EMO_MAP = {
    "negative": 0, "neg": 0, "neutral": 1, "neu": 1, "positive": 2, "pos": 2,
    "-1": 0, "-1.0": 0, -1: 0,
    "0": 1, "0.0": 1, 0: 1,
    "1": 2, "1.0": 2, 1: 2,
    "2": 2, "2.0": 2, 2: 2,
}
# Hỗ trợ cả 0/1 lẫn các mức 0, 1, 2, 3 (mức >=1 coi là toxic)
TOX_MAP = {
    "non-toxic": 0, "non_toxic": 0, "nontoxic": 0, "clean": 0,
    "toxic": 1,
    "0": 0, "0.0": 0, 0: 0,
    "1": 1, "1.0": 1, 1: 1,
    "2": 1, "2.0": 1, 2: 1,
    "3": 1, "3.0": 1, 3: 1,
}
SPLIT_MAP = {"train": "train", "val": "val", "valid": "val", "validation": "val", "dev": "val",
             "challenge": "challenge", "test": "challenge"}


# ----------------------------------------------------------------------------- dữ liệu
def parse_dataframe(df, text_col, emo_col, tox_col, split_col=None):
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    df[text_col] = df[text_col].fillna("").astype(str)

    # Map emotion
    ids_e = df[emo_col].astype(str).str.strip().str.lower().map(EMO_MAP)
    if ids_e.isna().any():
        bad = df.loc[ids_e.isna(), emo_col].unique()[:5]
        raise ValueError(f"Cột {emo_col}: có giá trị lạ {list(bad)} - cần sửa EMO_MAP")
    df["y_emotion"] = ids_e.astype(int)

    # Map toxicity
    ids_t = df[tox_col].astype(str).str.strip().str.lower().map(TOX_MAP)
    if ids_t.isna().any():
        bad = df.loc[ids_t.isna(), tox_col].unique()[:5]
        raise ValueError(f"Cột {tox_col}: có giá trị lạ {list(bad)} - cần sửa TOX_MAP")
    df["y_toxicity"] = ids_t.astype(int)

    if split_col and split_col in df.columns:
        df["split_"] = df[split_col].astype(str).str.strip().str.lower().map(SPLIT_MAP)
        if df["split_"].isna().any():
            raise ValueError(f"Cột {split_col} chỉ được chứa train/val/challenge, đang có: {df[split_col].unique()}")

    return df


def load_data(path, text_col, emo_col, tox_col, split_col):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return parse_dataframe(df, text_col, emo_col, tox_col, split_col)


def load_separate_files(tr_path, va_path, ch_path, text_col, emo_col, tox_col):
    dfs = []
    for p, sp in [(tr_path, "train"), (va_path, "val"), (ch_path, "challenge")]:
        if p and Path(p).exists():
            d = pd.read_csv(p, encoding="utf-8-sig")
            d["split_"] = sp
            dfs.append(d)
    if not dfs:
        raise ValueError("Không đọc được file train/val nào!")
    full = pd.concat(dfs, ignore_index=True)
    return parse_dataframe(full, text_col, emo_col, tox_col, "split_")


def build_vocab(texts, min_freq):
    cnt = Counter(w for t in texts for w in t.split())
    words = [w for w, c in cnt.items() if c >= min_freq]
    vocab = {"<pad>": 0, "<unk>": 1}
    for w in sorted(words, key=lambda w: (-cnt[w], w)):
        vocab[w] = len(vocab)
    return vocab, cnt


def encode(texts, vocab, max_len):
    X = np.zeros((len(texts), max_len), dtype="int32")
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()][:max_len]
        X[i, :len(ids)] = ids
    return X


# ----------------------------------------------------------------------------- fastText
def build_embedding_matrix(path, vocab, word_freq, dim, seed, out_dir):
    """Đọc file .vec theo dòng (không nạp cả file vào RAM), chỉ giữ từ có trong vocab.
    Từ ghép 'dịch_vụ' không có trong fastText -> lấy trung bình vector các âm tiết 'dịch','vụ'."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 0.1, (len(vocab), dim)).astype("float32")
    emb[0] = 0.0
    if not path or not Path(path).exists():
        print(f"!! Không tìm thấy --embeddings ({path}): dùng vector ngẫu nhiên (chỉ để test, kết quả sẽ thấp)")
        return emb

    syll = {p for w in vocab if "_" in w for p in w.split("_") if p}
    targets = set(vocab) | syll
    found = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        f.readline()                                        # dòng đầu: "<số từ> <số chiều>"
        for line in f:
            word, _, rest = line.partition(" ")
            if word in targets:
                v = np.asarray(rest.split(), dtype="float32")
                if v.shape[0] == dim:
                    found[word] = v
                    if len(found) == len(targets):
                        break

    direct = via_syll = oov = 0                             # đếm theo LƯỢT XUẤT HIỆN
    oov_words = []
    for w, idx in vocab.items():
        if idx < 2:
            continue
        f_ = word_freq.get(w, 1)
        if w in found:
            emb[idx] = found[w]; direct += f_
        elif "_" in w and any(p in found for p in w.split("_")):
            emb[idx] = np.mean([found[p] for p in w.split("_") if p in found], axis=0); via_syll += f_
        else:
            oov += f_; oov_words.append((w, f_))

    tot = direct + via_syll + oov
    print(f"Phủ fastText (theo lượt xuất hiện trong train): trực tiếp {direct/tot:.1%} | "
          f"qua âm tiết {via_syll/tot:.1%} | OOV {oov/tot:.1%}")
    oov_words.sort(key=lambda x: -x[1])
    (Path(out_dir) / "oov_top.txt").write_text(
        "\n".join(f"{w}\t{c}" for w, c in oov_words[:200]), encoding="utf-8")
    print("  -> 200 từ OOV phổ biến nhất lưu ở oov_top.txt")
    return emb


# ----------------------------------------------------------------------------- model
def build_model(emb, max_len, freeze, units, filters, dropout, lr):
    inp = keras.Input(shape=(max_len,), dtype="int32")
    x = layers.Embedding(emb.shape[0], emb.shape[1],
                         embeddings_initializer=keras.initializers.Constant(emb),
                         trainable=not freeze)(inp)
    x = layers.SpatialDropout1D(dropout)(x)
    x = layers.Bidirectional(layers.GRU(units, return_sequences=True))(x)   # Bi-GRU
    x = layers.LSTM(units, return_sequences=True)(x)                        # LSTM
    x = layers.Conv1D(filters, 3, padding="same", activation="relu")(x)     # CNN
    x = layers.Concatenate()([layers.GlobalMaxPooling1D()(x), layers.GlobalAveragePooling1D()(x)])
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = {"emotion": layers.Dense(3, activation="softmax", name="emotion")(x),
           "toxicity": layers.Dense(2, activation="softmax", name="toxicity")(x)}
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                   loss={"emotion": "sparse_categorical_crossentropy",
                         "toxicity": "sparse_categorical_crossentropy"})
    return model


def unpack(pred):
    if isinstance(pred, dict):
        return pred["emotion"], pred["toxicity"]
    return pred[0], pred[1]


class BestMacroF1(keras.callbacks.Callback):
    """Sau mỗi epoch: tính macro-F1 trên VAL, nhớ trọng số tốt nhất, tự giảm LR và dừng sớm khi hết cải thiện."""
    def __init__(self, X, ye, yt, patience, lr_patience=2, lr_factor=0.5, min_lr=1e-5):
        super().__init__()
        self.X, self.ye, self.yt = X, ye, yt
        self.patience, self.lr_patience, self.lr_factor, self.min_lr = patience, lr_patience, lr_factor, min_lr
        self.best, self.wait, self.best_w, self.best_epoch = -1.0, 0, None, 0
        self.wait_lr = 0

    def on_epoch_end(self, epoch, logs=None):
        pe, pt = unpack(self.model.predict(self.X, batch_size=256, verbose=0))
        fe = f1_score(self.ye, pe.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)
        ft = f1_score(self.yt, pt.argmax(1), average="macro", labels=[0, 1], zero_division=0)
        score = (fe + ft) / 2
        print(f"    val macro-F1: emotion={fe:.4f}  toxicity={ft:.4f}  trung bình={score:.4f}")

        if score > self.best + 1e-4:
            self.best = score
            self.best_w = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.wait = 0
            self.wait_lr = 0
        else:
            self.wait += 1
            self.wait_lr += 1
            # Tự động giảm Learning Rate nếu macro-F1 không tăng sau lr_patience epoch
            if self.wait_lr >= self.lr_patience:
                try:
                    curr_lr = float(self.model.optimizer.learning_rate.numpy())
                except Exception:
                    curr_lr = float(self.model.optimizer.learning_rate)
                if curr_lr > self.min_lr:
                    new_lr = max(curr_lr * self.lr_factor, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    print(f"    [Adaptive LR] Giảm Learning Rate: {curr_lr:.2e} -> {new_lr:.2e}")
                self.wait_lr = 0

            if self.wait >= self.patience:
                print(f"    Dừng sớm (Early Stopping) tại epoch {epoch + 1}")
                self.model.stop_training = True


# ----------------------------------------------------------------------------- đánh giá
def ece(probs, y, bins=10):
    conf, ok = probs.max(1), probs.argmax(1) == y
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def evaluate(model, df_split, X, name, out_dir):
    pe, pt = unpack(model.predict(X, batch_size=256, verbose=0))
    ye, yt = df_split["y_emotion"].values, df_split["y_toxicity"].values
    ae, at = pe.argmax(1), pt.argmax(1)
    res = {
        "n": int(len(ye)),
        "emotion": {"macro_f1": f1_score(ye, ae, average="macro", labels=[0, 1, 2], zero_division=0),
                    "ece": ece(pe, ye),
                    "report": classification_report(ye, ae, labels=[0, 1, 2], target_names=EMO_NAMES,
                                                    output_dict=True, zero_division=0),
                    "confusion": confusion_matrix(ye, ae, labels=[0, 1, 2]).tolist()},
        "toxicity": {"macro_f1": f1_score(yt, at, average="macro", labels=[0, 1], zero_division=0),
                     "ece": ece(pt, yt),
                     "report": classification_report(yt, at, labels=[0, 1], target_names=TOX_NAMES,
                                                     output_dict=True, zero_division=0),
                     "confusion": confusion_matrix(yt, at, labels=[0, 1]).tolist()},
    }
    fp = {}
    for i, en in enumerate(EMO_NAMES):
        m = (yt == 0) & (ye == i)
        fp[en] = {"n": int(m.sum()), "fp_rate": float((at[m] == 1).mean()) if m.any() else None}
    res["false_positive_rate_by_emotion"] = fp
    print(f"\n===== ĐÁNH GIÁ TẬP {name.upper()} (n={len(ye)}) =====")
    print(f"Emotion : macro-F1 {res['emotion']['macro_f1']:.4f} | ECE {res['emotion']['ece']:.4f}")
    print(f"Toxicity: macro-F1 {res['toxicity']['macro_f1']:.4f} | ECE {res['toxicity']['ece']:.4f}")
    print("Tỉ lệ báo nhầm toxic (trên bình luận non-toxic) theo cảm xúc:",
          {k: (None if v['fp_rate'] is None else round(v['fp_rate'], 3)) for k, v in fp.items()})
    print(classification_report(yt, at, labels=[0, 1], target_names=TOX_NAMES, zero_division=0))
    keep = [c for c in ["id", "text", "text_clean", "tox_level", "ambiguous", "annotator", "source"]
            if c in df_split.columns]
    out = df_split[keep].copy()
    out["y_emotion"], out["pred_emotion"] = ye, ae
    for i, en in enumerate(EMO_NAMES):
        out[f"p_{en}"] = pe[:, i]
    out["y_toxicity"], out["pred_toxicity"], out["p_toxic"] = yt, at, pt[:, 1]
    out.to_csv(Path(out_dir) / f"preds_{name}.csv", index=False, encoding="utf-8-sig")
    return res


# ----------------------------------------------------------------------------- dự đoán câu mới
def predict_text(run_dir, texts):
    from preprocess_fasttext import clean
    run = Path(run_dir)
    cfg = json.loads((run / "vocab.json").read_text(encoding="utf-8"))
    model = keras.saving.load_model(run / "model.keras")
    cleaned = [clean(t) for t in texts]
    X = encode(cleaned, cfg["vocab"], cfg["max_len"])
    pe, pt = unpack(model.predict(X, verbose=0))
    for t, c, e, k in zip(texts, cleaned, pe, pt):
        print(f"\n{t}\n  (sau xử lý: {c})")
        print(f"  Emotion : {EMO_NAMES[e.argmax()]:<9} {dict(zip(EMO_NAMES, [round(float(x), 3) for x in e]))}")
        print(f"  Toxicity: {TOX_NAMES[k.argmax()]:<9} P(toxic)={k[1]:.3f}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="đường dẫn file CSV gộp (chứa cả train/val)")
    ap.add_argument("--train-data", help="đường dẫn file train riêng (nếu không dùng --data)")
    ap.add_argument("--val-data", help="đường dẫn file val riêng (nếu không dùng --data)")
    ap.add_argument("--challenge-data", help="đường dẫn file challenge riêng (nếu có)")
    ap.add_argument("--embeddings", help="cc.vi.300.vec hoặc cc.vi.300.vec.gz")
    ap.add_argument("--text-col", default="text_clean")
    ap.add_argument("--emotion-col", default="emotion")
    ap.add_argument("--toxicity-col", default="toxicity")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default="bigru")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--max-len", type=int, default=0, help="0 = tự chọn theo phân vị 95")
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--filters", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--freeze", action="store_true", help="đóng băng embedding")
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--eval-challenge", action="store_true")
    ap.add_argument("--predict", help="thư mục run đã train")
    ap.add_argument("--text", action="append")
    a = ap.parse_args()

    if a.predict:
        predict_text(a.predict, a.text or ["dịch vụ tệ quá, chờ cả tiếng mới có món"])
        return

    if not a.data and not (a.train_data and a.val_data):
        ap.error("Cần cung cấp --data (file gộp) hoặc cả --train-data và --val-data")

    keras.utils.set_random_seed(a.seed)
    out_dir = Path(a.out_root) / f"{a.name}_seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.data:
        df = load_data(a.data, a.text_col, a.emotion_col, a.toxicity_col, a.split_col)
    else:
        df = load_separate_files(a.train_data, a.val_data, a.challenge_data, a.text_col, a.emotion_col, a.toxicity_col)

    tr = df[df.split_ == "train"].reset_index(drop=True)
    va = df[df.split_ == "val"].reset_index(drop=True)
    ch = df[df.split_ == "challenge"].reset_index(drop=True)
    print(f"Dataset nạp vào: train={len(tr)}  val={len(va)}  challenge={len(ch)}")
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("Thiếu dòng dữ liệu train hoặc val!")

    vocab, freq = build_vocab(tr[a.text_col], a.min_freq)
    lens = tr[a.text_col].map(lambda t: len(t.split()))
    max_len = a.max_len or int(np.clip(np.percentile(lens, 95), 20, 128))
    print(f"Kích thước từ điển vocab={len(vocab)} | max_len={max_len}")
    emb = build_embedding_matrix(a.embeddings, vocab, freq, 300, a.seed, out_dir)

    Xtr, Xva = encode(tr[a.text_col], vocab, max_len), encode(va[a.text_col], vocab, max_len)
    ytr = {"emotion": tr.y_emotion.values, "toxicity": tr.y_toxicity.values}
    sw = None
    if not a.no_class_weight:
        sw = {}
        for key, n in [("emotion", 3), ("toxicity", 2)]:
            cw = compute_class_weight("balanced", classes=np.arange(n), y=ytr[key])
            sw[key] = cw[ytr[key]].astype("float32")
        print("Class weight toxicity:", np.round(compute_class_weight("balanced", classes=np.arange(2),
                                                                     y=ytr["toxicity"]), 2))

    model = build_model(emb, max_len, a.freeze, a.units, a.filters, a.dropout, a.lr)
    model.summary(print_fn=lambda s: None)
    cb = BestMacroF1(Xva, va.y_emotion.values, va.y_toxicity.values, a.patience)
    model.fit(Xtr, ytr, sample_weight=sw, epochs=a.epochs, batch_size=a.batch_size, callbacks=[cb], verbose=2)
    if cb.best_w is not None:
        model.set_weights(cb.best_w)
    print(f"\nEpoch tốt nhất: {cb.best_epoch}  (val macro-F1 trung bình {cb.best:.4f})")

    metrics = {"config": vars(a), "best_epoch": cb.best_epoch, "val": evaluate(model, va, Xva, "val", out_dir)}
    if a.eval_challenge and len(ch):
        Xch = encode(ch[a.text_col], vocab, max_len)
        metrics["challenge"] = evaluate(model, ch, Xch, "challenge", out_dir)

    model.save(out_dir / "model.keras")
    (out_dir / "vocab.json").write_text(json.dumps({"vocab": vocab, "max_len": max_len}, ensure_ascii=False),
                                        encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
                                          encoding="utf-8")
    print(f"\nĐã lưu model + metrics + dự đoán vào: {out_dir}/")


if __name__ == "__main__":
    main()
