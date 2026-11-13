#!/usr/bin/env python3
"""
Tiền xử lý dataset cho nhánh  fastText 300d + Bi-GRU-LSTM-CNN  (đề tài NT120).

Đọc 1 file CSV có cột văn bản thô -> ghi ra CSV mới có thêm cột `text_clean`.

Thứ tự các bước (khác file mô tả của nhóm ở CHỖ ĐÁNH DẤU (*), lý do ghi ở cuối file):
    0. Chuẩn hóa Unicode NFC, thay URL
 (*)3. Emoticon gõ phím -> emoji  (giữ nguyên emoji, KHÔNG đổi thành emo_xxx)
    1. Rút gọn ký tự kéo dài  +  xóa ký tự chèn để lách luật (đ.m -> đm)
    2. Chuẩn hóa teencode / viết tắt / chửi thề
    6. Tách từ tiếng Việt (pyvi)
    4. Hạ chữ thường
    (bỏ bước 5 - ghép cụm phủ định - theo đúng yêu cầu của nhánh này)

Chạy thử:   python preprocess_fasttext.py --demo
Chạy thật:  python preprocess_fasttext.py --input data.csv --output data_ft.csv --text-col text
"""
import argparse
import re
import unicodedata

import emoji

# --------------------------------------------------------------------------
# Bước 3: emoticon gõ phím -> emoji   (bảng theo file mô tả của nhóm)
# Thứ tự QUAN TRỌNG: mẫu dài/đặc thù phải đứng trước mẫu ngắn.
# Emoticon có chữ cái (T_T, xD, :D, :v ...) bắt buộc đứng riêng, tránh dính vào từ thường.
# --------------------------------------------------------------------------
_B = r"(?<!\w)"      # không đứng sau chữ/số  (tránh "10:30" thành ":3")
_E = r"(?!\w)"       # không đứng trước chữ/số
EMOTICONS = [
    (r"</3", "💔"),
    (r"<3+", "❤"),
    (r"[:=]-?\){2,}", "😄"),            # :)) :))) =)) =)))
    (r";-?\)+", "😉"),                  # ;) ;-) ;))
    (r"[:=]-?\)", "🙂"),                # :) :-) =)
    (r"\^[_\-.]?\^", "😊"),             # ^^ ^_^ ^-^ ^.^
    (r"[:=]'-?\(", "😢"),               # :'( :'-(
    (r"[:=]-?\({2,}", "😢"),            # :(( :((( =(((
    (r"[:=]-?\(", "🙁"),                # :( :-( =(
    (r"-_-[\"']", "😓"),                # -_-"  -_-'
    (r"-_-|-\.-|;-;", "😑"),            # -_-  -.-  ;-;
    (r">_<", "😣"),
    (r"@@", "😵"),                      # (emoji "mắt xoắn ốc" là chuỗi ZWJ, pyvi sẽ tách vụn nên dùng 😵)
    (r";_;", "😭"),
    (_B + r"[Tt][_.][Tt]" + _E, "😭"),  # T_T T.T t_t
    (_B + r"[Xx]_[Xx]" + _E, "😵"),     # x_x X_X
    (_B + r"[Oo]_[Oo]" + _E, "😲"),     # O_O o_o
    (_B + r"[xX]-?[dD]" + _E, "😆"),    # xD XD x-D
    (_B + r"[:=]-?D" + _E, "😀"),       # :D =D :-D
    (_B + r"[:=]-?[Pp]" + _E, "😛"),    # :P :p =P
    (_B + r";-?[Pp]" + _E, "😜"),       # ;P ;p
    (_B + r":3" + _E, "🐱"),
    (_B + r"[:=]-?[Oo]" + _E, "😮"),    # :o :O
    (_B + r":-?[/\\](?![/\w])", "😕"),  # :/ :-/ :\
    (_B + r":S" + _E, "😕"),
    (r"[=:]\]{2,}", "😏"),              # =]]] :]]
    (_B + r":[vV]" + _E, "😏"),         # :v
]
EMOTICONS = [(re.compile(p), f" {e} ") for p, e in EMOTICONS]

# --------------------------------------------------------------------------
# Bước 2: teencode / viết tắt / chửi thề  ->  từ gốc.
# Đây là BẢN KHỞI ĐẦU. Hãy mở rộng bằng cách liệt kê các token xuất hiện nhiều
# nhất trong dữ liệu thật mà không có trong fastText (xem log OOV ở script train).
# --------------------------------------------------------------------------
TEENCODE = {
    # theo file mô tả của nhóm
    "k": "không", "ko": "không", "kg": "không", "khong": "không", "hok": "không", "hông": "không",
    "dc": "được", "đc": "được",
    "vs": "với", "r": "rồi", "rùi": "rồi",
    "mik": "mình", "mk": "mình", "ngta": "người ta", "vn": "việt nam",
    "a": "anh", "e": "em",          # theo ví dụ của nhóm - DỄ NHẦM (a = "à"?) -> nên đo lại F1
    "bt": "biết",                   # theo ví dụ của nhóm (cũng có thể là "bình thường")
    # thường gặp
    "j": "gì", "ntn": "như thế nào", "trc": "trước", "sp": "sản phẩm", "cx": "cũng", "cg": "cũng",
    "z": "vậy", "bik": "biết", "biet": "biết", "thik": "thích", "thix": "thích", "wa": "quá", "wá": "quá",
    "ns": "nói", "đt": "điện thoại", "ad": "admin",
    # chửi thề: gom các biến thể về 1 dạng (theo đề mục Bước 2)
    "vcl": "vãi cả lồn", "vkl": "vãi cả lồn", "vl": "vãi lồn", "vc": "vãi cả",
    "clgt": "cái lồn gì thế",
    "dm": "đm", "đmm": "đm", "dmm": "đm", "đm": "đm", "đjt": "đm",
}
_TEEN_RE = re.compile(
    r"(?<![\w\-])(" + "|".join(sorted(map(re.escape, TEENCODE), key=len, reverse=True)) + r")(?![\w\-])",
    flags=re.IGNORECASE,
)

# --------------------------------------------------------------------------
_L = r"[^\W\d_]"                                   # 1 chữ cái (kể cả có dấu)
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_LAUGH = re.compile(r"(?<!\w)(?:k{3,}|h{3,}|(?:ha|he|hi|hu){2,})(?!\w)", re.I)   # kkkk, hhhh, hihi
_EVADE = re.compile(rf"(?<!{_L})(?:{_L}[.\-*!_~|/\\+])+{_L}(?!{_L})")   # đ.m, l.ồ.n, d!m
_PUNCT = re.compile(r"([?!.,;])\1+")               # ???? -> ?   .... -> .
_HYPHEN = re.compile(r"(\w) - (\w)")               # pyvi tách "k-atm" thành "k - atm" -> nối lại
_SPACES = re.compile(r"\s+")

_segmenter = None


def _base(c: str) -> str:
    return unicodedata.normalize("NFD", c)[0].lower()


def collapse_repeats(t: str) -> str:
    """nguuuuu -> ngu ; quáaaaa -> quá  (chỉ áp dụng cho CHỮ CÁI, chuỗi >= 3 ký tự cùng chữ gốc).
    Không đụng tới dấu câu/emoticon (:))) khác :) )."""
    out, i = [], 0
    while i < len(t):
        c = t[i]
        if not c.isalpha():
            out.append(c)
            i += 1
            continue
        j, b = i + 1, _base(c)
        while j < len(t) and t[j].isalpha() and _base(t[j]) == b:
            j += 1
        run = t[i:j]
        if len(run) >= 3:      # giữ ký tự có dấu nếu có (quáaaa -> quá), không thì giữ ký tự đầu
            run = next((x for x in run if len(unicodedata.normalize("NFD", x)) > 1), run[0])
        out.append(run)
        i = j
    return "".join(out)


def segment(text: str) -> str:
    """Tách từ tiếng Việt bằng pyvi: 'dịch vụ' -> 'dịch_vụ'."""
    global _segmenter
    if _segmenter is None:
        from pyvi import ViTokenizer
        _segmenter = ViTokenizer
    return _segmenter.tokenize(text)


def clean(text, do_segment: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = _URL.sub(" url ", t)                                   # tránh ":/" trong link bị nhầm emoticon
    for pat, emo in EMOTICONS:                                 # bước 3 (làm sớm - xem ghi chú cuối file)
        t = pat.sub(emo, t)
    t = t.replace("\ufe0f", "")                                # bỏ variation selector: ❤️ == ❤
    t = emoji.replace_emoji(t, lambda e, d: f" {e} ")          # tách emoji ra khỏi chữ dính liền
    t = _LAUGH.sub(" haha ", t)                                # "kkkk" là cười, đừng để thành "k" = "không"
    t = collapse_repeats(t)                                    # bước 1a
    t = _EVADE.sub(lambda m: re.sub(r"[.\-*!_~|/\\+]", "", m.group(0)), t)   # bước 1b
    t = _PUNCT.sub(r"\1", t)
    t = _TEEN_RE.sub(lambda m: TEENCODE[m.group(1).lower()], t)   # bước 2
    t = _SPACES.sub(" ", t).strip()
    if do_segment and t:
        t = segment(t)                                         # bước 6
        t = _HYPHEN.sub(r"\1-\2", t)
    return _SPACES.sub(" ", t.lower()).strip()                 # bước 4


DEMO = [
    "So Pre Tài Trợ Độ Tây Ngày Hôm Ngày =))",
    "Bt các bạn phan cuồng chửi vậy mik đã ko nêu ra ý kiến cá nhân",
    "Panda Thái đã xem =)))",
    "Luôn ủng hộ K-ATM",
    "Giọng cuốn vc",
    "thế thua gì bác.. k công khai mà cả VN điều biết tên e nó bác nghĩ ngta đón được sao????",
    "Đồ ngu quáaaaa, đ.m mày, l.ồ.n thật :((  T_T",
    "Ok lúc 10:30 nha :3   https://vnexpress.net/abc  kkkkk",
    "Dịch vụ tệ quá 😡 ❤️ ^^",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="CSV đầu vào")
    ap.add_argument("--output", help="CSV đầu ra (thêm cột text_clean)")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--no-segment", action="store_true",
                    help="bỏ bước tách từ (để so OOV với fastText có/không tách từ)")
    ap.add_argument("--demo", action="store_true", help="in ví dụ trước/sau rồi thoát")
    a = ap.parse_args()

    if a.demo or not a.input:
        for s in DEMO:
            print(f"GỐC : {s}\nSẠCH: {clean(s, not a.no_segment)}\n")
        return

    import pandas as pd
    df = pd.read_csv(a.input)
    df["text_clean"] = df[a.text_col].map(lambda x: clean(x, not a.no_segment))
    empty = (df["text_clean"] == "").sum()
    df.to_csv(a.output, index=False, encoding="utf-8-sig")
    print(f"Xong {len(df)} dòng -> {a.output}  (dòng rỗng sau khi làm sạch: {empty})")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# (*) Vì sao khác file mô tả của nhóm?
# File nhóm ghi "bước 6 (tách từ) trước bước 3 (emoticon)". Khi thử thì pyvi BĂM emoticon:
#     "=)))"  ->  "= ) ) )"      "T_T ^^ <3"  ->  "T_T ^ ^ < 3"
# nên nếu tách từ trước thì không còn nhận ra emoticon nữa. Vì vậy phải đổi emoticon -> emoji
# TRƯỚC khi tách từ (pyvi giữ nguyên emoji). Kết quả cuối vẫn đúng yêu cầu: giữ emoji, không emo_xxx.
# Cũng vì lý do đó bước 1 (rút gọn/xóa ký tự chèn) làm SAU emoticon: "T.T" nếu bị xóa dấu chấm
# sẽ thành "TT", còn ":)))" nếu bị rút gọn sẽ mất nghĩa.
# ---------------------------------------------------------------------------
