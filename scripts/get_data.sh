#!/usr/bin/env bash
# 一鍵：下載三個公開數據集 -> 轉成 YOLO 格式 -> 驗證。
# 產出 data/yolo/（可直接訓練）。資料集不重新散布，皆從官方來源抓。
#
# 前置：
#   pip install -r requirements.txt
#   FLIR / KAIST 需 Kaggle API token：
#     ~/.kaggle/kaggle.json  或  export KAGGLE_USERNAME=.. KAGGLE_KEY=..
#   （只想要 LLVIP 免認證：跑 `python scripts/download.py --llvip`）
#
# 用法：bash scripts/get_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/2] 下載數據集 -> data/raw/"
python scripts/download.py --all

echo "[2/2] 轉換 + 驗證 -> data/yolo/"
bash scripts/prepare_all.sh

echo "完成。開訓：python scripts/train.py --device 0"
