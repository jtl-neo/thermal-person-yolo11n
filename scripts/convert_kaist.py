"""KAIST Multispectral Pedestrian -> YOLO 格式。使用 LWIR（熱影像）+ bbGt txt 標註。

預期目錄（標準釋出）:
  KAIST/
    images/setXX/VYYY/lwir/I#####.jpg
    annotations/setXX/VYYY/I#####.txt   (bbGt version=3)

bbGt 每列: <label> <x> <y> <w> <h> <occ> <xl> <yl> <wl> <hl> <ignore> <ang>
  x y w h = 左上角像素座標。

只保留 --classes 指定類別（預設 person），全部對應 class id 0。
用法:
  python scripts/convert_kaist.py --src data/raw/KAIST --dst data/yolo
  # 納入 cyclist:
  python scripts/convert_kaist.py --src data/raw/KAIST --dst data/yolo --classes person cyclist
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from common import (
    ensure_yolo_dirs,
    fmt_line,
    hashed_split,
    place_image,
    valid_box,
    write_label,
    xywh_to_yolo,
)


def find_lwir_images(src: Path) -> list[Path]:
    imgs = [p for p in src.rglob("*.jpg") if "lwir" in p.parts or "lwir" in p.parent.name.lower()]
    if not imgs:  # fallback: 有些釋出 thermal 資料夾名不同
        imgs = [p for p in src.rglob("*.jpg") if any(k in str(p).lower() for k in ("lwir", "thermal", "ir"))]
    return sorted(imgs)


def find_ann(src: Path, img: Path) -> Path | None:
    # images/setXX/VYYY/lwir/I#####.jpg -> annotations/setXX/VYYY/I#####.txt
    parts = list(img.parts)
    stem = img.stem
    for anndir in ("annotations", "annotations-xml", "labels"):
        # 嘗試把 'images' 換成 anndir 並移除 'lwir'
        try:
            idx = parts.index("images")
        except ValueError:
            idx = None
        if idx is not None:
            rel = [p for p in parts[idx + 1 :] if p.lower() != "lwir"]
            cand = src / anndir / Path(*rel[:-1]) / f"{stem}.txt"
            if cand.exists():
                return cand
    hits = list(src.rglob(f"{stem}.txt"))
    return hits[0] if hits else None


def parse_bbgt(txt: Path, keep: set[str]) -> list[tuple]:
    boxes = []
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        label = parts[0].lower()
        if label not in keep:
            continue
        try:
            x, y, w, h = (float(v) for v in parts[1:5])
        except (ValueError, IndexError):
            continue
        boxes.append((x, y, w, h))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--classes", nargs="+", default=["person"],
                    help="要保留的 KAIST 類別（全部映射為 person=0）")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--prefix", default="kaist")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"[KAIST] src 不存在: {args.src}")
    ensure_yolo_dirs(args.dst)
    keep = {c.lower() for c in args.classes}

    imgs = find_lwir_images(args.src)
    if not imgs:
        sys.exit(f"[KAIST] 找不到 LWIR 熱影像於 {args.src}")

    n_ok = n_box = n_skip = 0
    for img in imgs:
        ann = find_ann(args.src, img)
        if ann is None:
            n_skip += 1
            continue
        try:
            w, h = Image.open(img).size
        except OSError:
            n_skip += 1
            continue
        lines = []
        for x, y, bw_, bh_ in parse_bbgt(ann, keep):
            cx, cy, bw, bh = xywh_to_yolo(x, y, bw_, bh_, w, h)
            if valid_box(bw, bh):
                lines.append(fmt_line(0, cx, cy, bw, bh))
        # 用相對路徑組唯一 stem，避免不同 set 撞名
        rel = img.relative_to(args.src).with_suffix("")
        uid = "_".join(rel.parts).replace("lwir_", "")
        stem = f"{args.prefix}_{uid}"
        split = hashed_split(f"{args.prefix}/{uid}", args.val, args.test)
        place_image(img, args.dst / "images" / split, f"{stem}{img.suffix.lower()}", args.copy)
        write_label(args.dst / "labels" / split, stem, lines)
        n_ok += 1
        n_box += len(lines)

    print(f"[KAIST] 影像 {n_ok}，box {n_box}，略過 {n_skip}（無標註/讀取失敗）")


if __name__ == "__main__":
    main()
