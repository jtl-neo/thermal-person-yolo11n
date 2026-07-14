"""下載三個數據集到 data/raw/。

來源:
  LLVIP  — Google Drive（官方，需 gdown；資料集 ~4GB）
  FLIR   — Kaggle: deepnewbie/flir-thermal-images-dataset（需 Kaggle API token）
  KAIST  — Kaggle: adlteam/kaist-dataset（需 Kaggle API token）

Kaggle 認證（FLIR/KAIST 必須）:
  1. kaggle.com -> Account -> Create New API Token 下載 kaggle.json
  2. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
     或設環境變數 KAGGLE_USERNAME / KAGGLE_KEY

用法:
  python scripts/download.py --all
  python scripts/download.py --llvip
  python scripts/download.py --flir --kaist
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

RAW = Path("data/raw")
LLVIP_GDRIVE_ID = "1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv"  # 官方 LLVIP.zip


def link_or_copy_tree(src: Path, dst: Path):
    """kagglehub 下載到 cache，這裡建 symlink 指過去（省空間）。"""
    if dst.exists() or dst.is_symlink():
        print(f"  已存在，跳過: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(Path(src).resolve(), dst)
        print(f"  symlink {dst} -> {src}")
    except OSError:
        shutil.copytree(src, dst)
        print(f"  copied -> {dst}")


def dl_llvip():
    print("[LLVIP] Google Drive 下載…")
    try:
        import gdown
    except ImportError:
        sys.exit("需要 gdown: pip install gdown")
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "LLVIP.zip"
    if not zpath.exists():
        gdown.download(id=LLVIP_GDRIVE_ID, output=str(zpath), quiet=False)
    else:
        print(f"  zip 已存在: {zpath}")
    out = RAW / "LLVIP"
    if out.exists():
        print(f"  已解壓: {out}")
        return
    print("  解壓中…")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW)
    # 官方 zip 頂層通常就是 LLVIP/；若名稱不同自行確認
    print(f"[LLVIP] 完成 -> {out}（converter 期望 infrared/ + Annotations/）")


def dl_kaggle(slug: str, dst_name: str):
    print(f"[Kaggle] {slug} 下載…")
    try:
        import kagglehub
    except ImportError:
        sys.exit("需要 kagglehub: pip install kagglehub")
    path = kagglehub.dataset_download(slug)  # 下載到 ~/.cache/kagglehub 並回傳路徑
    print(f"  cache: {path}")
    link_or_copy_tree(Path(path), RAW / dst_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--llvip", action="store_true")
    ap.add_argument("--flir", action="store_true")
    ap.add_argument("--kaist", action="store_true")
    args = ap.parse_args()
    if not any([args.all, args.llvip, args.flir, args.kaist]):
        ap.print_help()
        sys.exit(0)

    if args.all or args.llvip:
        dl_llvip()
    if args.all or args.flir:
        dl_kaggle("deepnewbie/flir-thermal-images-dataset", "FLIR")
    if args.all or args.kaist:
        dl_kaggle("adlteam/kaist-dataset", "KAIST")

    print("\n下一步: bash scripts/prepare_all.sh")


if __name__ == "__main__":
    main()
