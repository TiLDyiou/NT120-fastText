"""Environment verification script for Phase 0."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import platform
import sys
from pathlib import Path

from src_v2.config import DEFAULT_FASTTEXT_BIN, RAW_DATA_DIR, ensure_project_dirs


def get_ram_gb() -> tuple[float, float]:
    """Return (total_gb, available_gb) on Windows or Linux."""
    if sys.platform == "win32":
        class MEMSTAT(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / (1024 ** 3), m.ullAvailPhys / (1024 ** 3)
    return 0.0, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 0 runtime dependencies.")
    parser.add_argument("--bin", type=Path, default=DEFAULT_FASTTEXT_BIN, help="Path to cc.vi.300.bin")
    parser.add_argument("--load-bin", action="store_true", help="Actually load cc.vi.300.bin and test 'xin'")
    args = parser.parse_args()

    ensure_project_dirs()
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"OS: {platform.platform()}")
    total_ram, avail_ram = get_ram_gb()
    if total_ram > 0:
        print(f"RAM: total={total_ram:.1f} GB, available={avail_ram:.1f} GB")

    required = ["torch", "numpy", "pandas", "sklearn", "scipy", "tqdm", "pyarrow", "pytest"]
    missing = []
    for pkg in required:
        ok = importlib.util.find_spec(pkg) is not None
        print(f"  pkg {pkg:<10}: {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(pkg)

    ft_ok = importlib.util.find_spec("fasttext") is not None
    gensim_ok = importlib.util.find_spec("gensim") is not None
    print(f"  pkg fasttext  : {'OK' if ft_ok else 'MISSING'}")
    print(f"  pkg gensim    : {'OK' if gensim_ok else 'MISSING (optional fallback)'}")

    import torch
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    for split in ["train", "valid", "challenge"]:
        p = RAW_DATA_DIR / split / f"{split}.csv"
        print(f"  raw {split:<9}: {'OK' if p.exists() else 'MISSING'} -> {p}")

    print(f"  fastText .bin : {'OK' if args.bin.exists() else 'MISSING'} -> {args.bin}")
    if args.load_bin:
        if not args.bin.exists():
            raise SystemExit(f"Missing fastText .bin at {args.bin}")
        if ft_ok:
            import fasttext
            m = fasttext.load_model(str(args.bin))
            vec = m.get_word_vector("xin")
            print(f"  fasttext.get_word_vector('xin'): shape={vec.shape}, norm={float((vec**2).sum()**0.5):.4f}")
        elif gensim_ok:
            from gensim.models.fasttext import load_facebook_vectors
            m = load_facebook_vectors(str(args.bin))
            vec = m["xin"]
            print(f"  gensim load_facebook_vectors('xin'): shape={vec.shape}, norm={float((vec**2).sum()**0.5):.4f}")
        else:
            raise SystemExit("Neither fasttext nor gensim is installed.")

    if missing or (not ft_ok and not gensim_ok):
        raise SystemExit("Environment check failed due to missing packages.")
    print("Phase 0 environment check PASSED.")


if __name__ == "__main__":
    main()
