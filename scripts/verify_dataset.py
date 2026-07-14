"""檢查轉換後的 YOLO 數據集：計數、label 格式、image/label 配對、抽樣座標範圍。

用法:
  python scripts/verify_dataset.py --dst data/yolo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png")


def check_split(dst: Path, split: str) -> tuple[int, int, int, list[str]]:
    img_dir = dst / "images" / split
    lbl_dir = dst / "labels" / split
    errs: list[str] = []
    imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT]
    n_box = 0
    n_missing_lbl = 0
    for img in imgs:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            n_missing_lbl += 1
            continue
        for i, line in enumerate(lbl.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errs.append(f"{lbl}:{i} 欄位數 != 5")
                continue
            cls, *coords = parts
            if cls != "0":
                errs.append(f"{lbl}:{i} class != 0 ({cls})")
            for c in coords:
                v = float(c)
                if not (0.0 <= v <= 1.0):
                    errs.append(f"{lbl}:{i} 座標超出 [0,1]: {c}")
            n_box += 1
    return len(imgs), n_box, n_missing_lbl, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=Path, default=Path("data/yolo"))
    ap.add_argument("--max-errs", type=int, default=20)
    args = ap.parse_args()

    total_img = total_box = 0
    all_errs: list[str] = []
    print(f"{'split':<6} {'images':>8} {'boxes':>8} {'no-label':>9}")
    for split in ("train", "val", "test"):
        n_img, n_box, n_miss, errs = check_split(args.dst, split)
        total_img += n_img
        total_box += n_box
        all_errs += errs
        print(f"{split:<6} {n_img:>8} {n_box:>8} {n_miss:>9}")
    print(f"{'總計':<6} {total_img:>8} {total_box:>8}")

    if total_img == 0:
        sys.exit("錯誤: 沒有任何影像，先跑 convert_*.py")
    if all_errs:
        print(f"\n發現 {len(all_errs)} 個格式問題（顯示前 {args.max_errs}）:")
        for e in all_errs[: args.max_errs]:
            print("  ", e)
        sys.exit(1)
    print("\nOK: 格式檢查通過，可開始訓練。")


if __name__ == "__main__":
    main()
