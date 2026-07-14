#!/usr/bin/env bash
# 一鍵轉換三個數據集 -> data/yolo，然後驗證。
# 先把資料下載解壓到 data/raw/{LLVIP,FLIR,KAIST}。缺哪個就跳過哪個。
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=data/raw
DST=data/yolo

run() { echo "+ $*"; python "$@"; }

[ -d "$RAW/LLVIP" ] && run scripts/convert_llvip.py --src "$RAW/LLVIP" --dst "$DST" || echo "跳過 LLVIP（$RAW/LLVIP 不存在）"
[ -d "$RAW/FLIR" ]  && run scripts/convert_flir.py  --src "$RAW/FLIR"  --dst "$DST" || echo "跳過 FLIR（$RAW/FLIR 不存在）"
[ -d "$RAW/KAIST" ] && run scripts/convert_kaist.py --src "$RAW/KAIST" --dst "$DST" || echo "跳過 KAIST（$RAW/KAIST 不存在）"

run scripts/verify_dataset.py --dst "$DST"
echo "資料就緒。開訓: python scripts/train.py"
