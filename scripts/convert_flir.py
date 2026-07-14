"""FLIR ADAS -> YOLO 格式。使用熱影像 + COCO JSON 標註，只保留 person。

FLIR ADAS 版本差異大，此腳本自動搜尋 COCO json 與影像資料夾。
新版結構常為:
  FLIR_ADAS_v2/
    images_thermal_train/
      coco.json
      data/*.jpg
    images_thermal_val/
      coco.json
      data/*.jpg

用法:
  python scripts/convert_flir.py --src data/raw/FLIR --dst data/yolo
  # 或明確指定:
  python scripts/convert_flir.py --ann .../coco.json --img-dir .../data --dst data/yolo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    ensure_yolo_dirs,
    fmt_line,
    hashed_split,
    place_image,
    valid_box,
    write_label,
    xywh_to_yolo,
)


def find_coco_jsons(src: Path) -> list[Path]:
    names = ("coco.json", "instances.json", "index.json", "thermal_annotations.json")
    hits = []
    for p in src.rglob("*.json"):
        if p.name.lower() in names or "coco" in p.name.lower() or "annot" in p.name.lower():
            hits.append(p)
    return sorted(set(hits))


def person_cat_ids(cats: list[dict]) -> set[int]:
    ids = set()
    for c in cats:
        if "person" in str(c.get("name", "")).lower():
            ids.add(c["id"])
    return ids


def resolve_img_dir(ann_path: Path, img_dir: Path | None) -> Path:
    if img_dir:
        return img_dir
    for cand in ("data", "thermal", "."):
        d = ann_path.parent / cand
        if d.is_dir() and any(d.glob("*.jp*g")):
            return d
    return ann_path.parent


def convert_one(ann_path: Path, img_dir: Path | None, dst: Path, prefix: str,
                val: float, test: float, copy: bool) -> tuple[int, int, int]:
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    img_dir = resolve_img_dir(ann_path, img_dir)
    pids = person_cat_ids(coco.get("categories", []))
    if not pids:
        print(f"[FLIR] 警告: {ann_path} 找不到 person category，略過")
        return 0, 0, 0

    images = {im["id"]: im for im in coco["images"]}
    boxes_by_img: dict[int, list] = {}
    for a in coco.get("annotations", []):
        if a.get("category_id") in pids and not a.get("iscrowd", 0):
            boxes_by_img.setdefault(a["image_id"], []).append(a["bbox"])

    n_ok = n_box = n_skip = 0
    for iid, im in images.items():
        fname = Path(im["file_name"]).name
        src_img = img_dir / fname
        if not src_img.exists():
            hit = next(iter(img_dir.rglob(fname)), None)
            if hit is None:
                n_skip += 1
                continue
            src_img = hit
        w, h = im.get("width"), im.get("height")
        if not w or not h:
            n_skip += 1
            continue
        lines = []
        for x, y, bw_, bh_ in boxes_by_img.get(iid, []):
            cx, cy, bw, bh = xywh_to_yolo(x, y, bw_, bh_, w, h)
            if valid_box(bw, bh):
                lines.append(fmt_line(0, cx, cy, bw, bh))
        stem = Path(fname).stem
        split = hashed_split(f"{prefix}/{stem}", val, test)
        out_name = f"{prefix}_{stem}"
        place_image(src_img, dst / "images" / split, f"{out_name}{src_img.suffix.lower()}", copy)
        write_label(dst / "labels" / split, out_name, lines)
        n_ok += 1
        n_box += len(lines)
    return n_ok, n_box, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, help="FLIR 根目錄（自動搜尋 coco.json）")
    ap.add_argument("--ann", type=Path, help="明確指定單一 coco.json")
    ap.add_argument("--img-dir", type=Path, help="搭配 --ann 的影像資料夾")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--prefix", default="flir")
    args = ap.parse_args()

    ensure_yolo_dirs(args.dst)

    if args.ann:
        anns = [args.ann]
    elif args.src:
        if not args.src.is_dir():
            sys.exit(f"[FLIR] src 不存在: {args.src}")
        anns = find_coco_jsons(args.src)
        if not anns:
            sys.exit(f"[FLIR] 找不到 COCO json 於 {args.src}")
    else:
        sys.exit("[FLIR] 需提供 --src 或 --ann")

    tot = [0, 0, 0]
    for ann in anns:
        print(f"[FLIR] 處理 {ann}")
        r = convert_one(ann, args.img_dir, args.dst, args.prefix, args.val, args.test, args.copy)
        tot = [a + b for a, b in zip(tot, r)]
    print(f"[FLIR] 影像 {tot[0]}，box {tot[1]}，略過 {tot[2]}")


if __name__ == "__main__":
    main()
