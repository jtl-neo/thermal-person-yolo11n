"""同一批 IR test 圖，原版 vs finetuned 並排標框對比。

用法:
  python scripts/compare_predict.py --n 6 --conf 0.35
輸出: runs/compare/*.jpg（左=stock COCO, 右=finetuned）
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import cv2
from ultralytics import YOLO

random.seed(0)


def draw(img, boxes, color, tag):
    img = img.copy()
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, f"{c:.2f}", (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, f"{tag}  ({len(boxes)} person)", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def preds(model, path, conf, cls_filter):
    r = model.predict(path, conf=conf, classes=cls_filter, verbose=False, device=0)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--test-dir", default="data/yolo/images/test")
    ap.add_argument("--label-dir", default="data/yolo/labels/test")
    ap.add_argument("--out", default="runs/compare")
    args = ap.parse_args()

    stock = YOLO("yolo11n.pt")
    fine = YOLO("runs/detect/runs/ir_person_yolo11n/weights/best.pt")

    # 挑有 person 的圖（label 非空），並混不同來源（llvip/flir）
    lbls = list(Path(args.label_dir).glob("*.txt"))
    withp = [p for p in lbls if p.stat().st_size > 0]
    random.shuffle(withp)
    picks, seen_pref = [], set()
    for p in withp:
        pref = p.stem.split("_")[0]
        # 盡量各來源都取
        if len([x for x in picks if x.stem.startswith(pref)]) < args.n // 2 + 1:
            picks.append(p)
        if len(picks) >= args.n:
            break

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for lbl in picks:
        stem = lbl.stem
        img_path = next(Path(args.test_dir).glob(f"{stem}.*"))
        img = cv2.imread(str(img_path))
        left = draw(img, preds(stock, str(img_path), args.conf, [0]), (0, 165, 255), "STOCK yolo11n")
        right = draw(img, preds(fine, str(img_path), args.conf, [0]), (0, 255, 0), "FINETUNED")
        sep = 255 * (img[:, :4] * 0 + 1)
        combo = cv2.hconcat([left, sep, right])
        cv2.imwrite(str(out / f"{stem}.jpg"), combo)
        print("saved", out / f"{stem}.jpg")


if __name__ == "__main__":
    main()
