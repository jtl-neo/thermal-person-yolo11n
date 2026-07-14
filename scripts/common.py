"""共用工具：確定性 split、影像放置、YOLO 標註寫入、bbox 轉換。"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")


def hashed_split(key: str, val_ratio: float, test_ratio: float) -> str:
    """依檔名雜湊做確定性切分，跨資料集/多次執行結果一致。"""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF  # 0..1
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def ensure_yolo_dirs(dst: Path) -> None:
    for split in SPLITS:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)


def place_image(src: Path, dst_dir: Path, out_name: str, copy: bool = False) -> None:
    """symlink（預設，省空間）或 copy 影像到目的 split 目錄。"""
    dst = dst_dir / out_name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def write_label(dst_dir: Path, stem: str, lines: list[str]) -> None:
    """寫 YOLO label（可能為空 = 無目標的負樣本）。"""
    (dst_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def voc_to_yolo(xmin, ymin, xmax, ymax, w, h):
    """VOC 像素 box -> YOLO 正規化 (cx, cy, bw, bh)。"""
    xmin, xmax = sorted((float(xmin), float(xmax)))
    ymin, ymax = sorted((float(ymin), float(ymax)))
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return _clamp01(cx), _clamp01(cy), _clamp01(bw), _clamp01(bh)


def xywh_to_yolo(x, y, bw, bh, w, h):
    """COCO/KAIST 左上角像素 (x,y,w,h) -> YOLO 正規化。"""
    cx = (float(x) + float(bw) / 2.0) / w
    cy = (float(y) + float(bh) / 2.0) / h
    return _clamp01(cx), _clamp01(cy), _clamp01(float(bw) / w), _clamp01(float(bh) / h)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def fmt_line(cls: int, cx, cy, bw, bh) -> str:
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def valid_box(bw, bh, min_size: float = 1e-3) -> bool:
    return bw > min_size and bh > min_size
