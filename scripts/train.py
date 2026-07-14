"""訓練 yolo11n on IR person。針對熱影像關閉色彩增強、保留幾何增強。

用法:
  python scripts/train.py                       # 預設 100 epochs
  python scripts/train.py --epochs 150 --batch 32
  python scripts/train.py --resume runs/ir_person_yolo11n/weights/last.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="起始權重（COCO 預訓練）")
    ap.add_argument("--data", default="configs/ir_person.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0 或 0,1 或 cpu")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="ir_person_yolo11n")
    ap.add_argument("--resume", default=None, help="從 last.pt 續訓")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # --- IR/熱影像增強策略 ---
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,   # 幾乎關色彩，保留亮度抖動
        fliplr=0.5, flipud=0.0,
        degrees=0.0, translate=0.1, scale=0.5,
        mosaic=1.0, close_mosaic=10,       # 最後 10 epoch 關 mosaic
        patience=30,
    )
    print(f"完成。權重: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
