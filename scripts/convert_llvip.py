"""LLVIP -> YOLO 格式。使用紅外線影像 + PASCAL VOC XML 標註（只有 person）。

預期 LLVIP 目錄（下載解壓後）:
  LLVIP/
    infrared/
      train/*.jpg   (或直接 *.jpg)
      test/*.jpg
    Annotations/*.xml

用法:
  python scripts/convert_llvip.py --src data/raw/LLVIP --dst data/yolo
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from common import (
    ensure_yolo_dirs,
    fmt_line,
    hashed_split,
    place_image,
    valid_box,
    voc_to_yolo,
    write_label,
)


def find_ir_images(src: Path) -> list[Path]:
    ir_root = None
    for cand in ("infrared", "IR", "ir"):
        if (src / cand).is_dir():
            ir_root = src / cand
            break
    if ir_root is None:
        ir_root = src
    imgs = sorted(p for p in ir_root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return imgs


def find_ann(src: Path, stem: str) -> Path | None:
    for cand in ("Annotations", "annotations"):
        p = src / cand / f"{stem}.xml"
        if p.exists():
            return p
    hits = list(src.rglob(f"{stem}.xml"))
    return hits[0] if hits else None


def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width")))
    h = int(float(size.findtext("height")))
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name != "person":
            continue
        b = obj.find("bndbox")
        boxes.append((b.findtext("xmin"), b.findtext("ymin"), b.findtext("xmax"), b.findtext("ymax")))
    return w, h, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--copy", action="store_true", help="複製影像而非 symlink")
    ap.add_argument("--prefix", default="llvip", help="輸出檔名前綴，避免跨資料集撞名")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"[LLVIP] src 不存在: {args.src}")
    ensure_yolo_dirs(args.dst)

    imgs = find_ir_images(args.src)
    if not imgs:
        sys.exit(f"[LLVIP] 找不到紅外線影像於 {args.src}")

    n_ok = n_box = n_skip = 0
    for img in imgs:
        stem = img.stem
        ann = find_ann(args.src, stem)
        if ann is None:
            n_skip += 1
            continue
        w, h, boxes = parse_voc(ann)
        if w <= 0 or h <= 0:
            n_skip += 1
            continue
        lines = []
        for xmin, ymin, xmax, ymax in boxes:
            cx, cy, bw, bh = voc_to_yolo(xmin, ymin, xmax, ymax, w, h)
            if valid_box(bw, bh):
                lines.append(fmt_line(0, cx, cy, bw, bh))
        split = hashed_split(f"{args.prefix}/{stem}", args.val, args.test)
        out_name = f"{args.prefix}_{stem}"
        place_image(img, args.dst / "images" / split, f"{out_name}{img.suffix.lower()}", args.copy)
        write_label(args.dst / "labels" / split, out_name, lines)
        n_ok += 1
        n_box += len(lines)

    print(f"[LLVIP] 影像 {n_ok}，box {n_box}，略過 {n_skip}（無標註）")


if __name__ == "__main__":
    main()
