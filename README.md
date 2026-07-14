# ir_yolo

微調 **YOLO11n**，用於**紅外線（IR / 熱影像）下的行人（person）偵測**，使用公開紅外線數據集：**LLVIP**、**FLIR ADAS**、**KAIST 多光譜行人數據集**。

以 RGB 預訓練的偵測器在熱影像上表現會退化（沒有色彩、對比反轉、紋理統計不同）。本專案將輕量的 `yolo11n` backbone 在紅外線資料上微調，讓它在夜間、煙霧、低能見度環境下仍能穩定偵測 `person` 類別。

---

## 功能

- 專為 IR / 熱影像優化的單一 `person` 類別偵測器。
- 將 LLVIP / FLIR ADAS / KAIST 轉換成統一的 YOLO 格式數據集。
- 可重現的 `yolo11n` 訓練設定（Ultralytics）。
- 在保留的 IR 測試集上評估（mAP、precision、recall）。
- 匯出 ONNX / TensorRT，用於邊緣部署。

---

## 數據集

| 數據集 | 模態 | 內容 | 連結 |
|---|---|---|---|
| **LLVIP** | 可見光 + 紅外線，成對 | 約 15k 對齊的 RGB/IR 影像，街景，夜間行人 | https://github.com/bupt-ai-cz/LLVIP |
| **FLIR ADAS** | 熱影像 | 1 萬多張熱影像，行車場景（person / car / bicycle） | https://www.flir.com/oem/adas/adas-dataset-form/ |
| **KAIST** | 可見光 + 紅外線，成對 | 約 95k 對齊的彩色-熱影像，校園/都市行人 | https://github.com/SoonminHwang/rgbt-ped-detection |

> **授權注意：** 各數據集有各自的授權與使用條款。請從官方來源註冊/下載並遵守其條款。本 repo **不**重新散布數據集。

本任務只使用 **IR / 熱影像**串流。RGB 成對影像（LLVIP、KAIST）訓練時忽略。

---

## 環境需求

```bash
python >= 3.9
pip install ultralytics
# 選用：匯出
pip install onnx onnxruntime
```

強烈建議使用 GPU（CUDA）。`yolo11n` 很小，單張中階 GPU 即可訓練（本專案在 RTX 4060 Ti 16GB 上測試）。

> **CUDA 版本對齊**：`pip install torch` 預設可能抓到比驅動更新的 CUDA build（如 cu130），導致 `torch.cuda.is_available()` 為 `False` 掉回 CPU。用 `nvidia-smi` 看驅動支援的 CUDA 版本再裝對應 torch。例：驅動 CUDA 12.6 →
> ```bash
> pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu126
> ```
> 驗證：`python -c "import torch; print(torch.cuda.is_available())"` 應為 `True`。

---

## 目錄結構

```
ir_yolo/
├── data/
│   ├── raw/                # 下載的原始數據集（LLVIP, FLIR, KAIST）
│   └── yolo/               # 轉換後的統一 YOLO 格式數據集
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── scripts/
│   ├── common.py           # 共用工具（split、bbox 轉換）
│   ├── convert_llvip.py    # LLVIP  VOC XML -> YOLO
│   ├── convert_flir.py     # FLIR   COCO   -> YOLO
│   ├── convert_kaist.py    # KAIST  bbGt   -> YOLO
│   ├── verify_dataset.py   # 檢查計數與 label 格式
│   ├── train.py            # yolo11n 訓練包裝
│   └── prepare_all.sh      # 一鍵轉換 + 驗證
├── configs/
│   └── ir_person.yaml      # Ultralytics 數據設定檔
├── runs/                   # 訓練輸出（權重、log）
├── requirements.txt
└── README.md
```

---

## 快速開始

```bash
# 1) 安裝依賴
pip install -r requirements.txt

# 2) 一鍵取得資料集（下載 + 轉 YOLO + 驗證 → data/yolo/）
#    FLIR/KAIST 需先設 Kaggle token（見「資料下載」）
bash scripts/get_data.sh

# 3) 開始訓練
python scripts/train.py --device 0
```

**只想推論不用重訓**：權重已在 repo `models/`（`.pt` + `.onnx`），直接用：

```bash
yolo detect predict model=models/ir_person_yolo11n.pt source=<ir_image> device=0
```

---

## 資料下載

用 `scripts/download.py` 一鍵下載到 `data/raw/`。

```bash
python scripts/download.py --all          # 三個都抓
python scripts/download.py --llvip        # 只抓 LLVIP
python scripts/download.py --flir --kaist # 只抓 Kaggle 兩個
```

來源與需求：
- **LLVIP** — 官方 Google Drive（`gdown`，~4GB zip → 15488 對影像）。無需認證。
- **FLIR** — Kaggle `deepnewbie/flir-thermal-images-dataset`（~15GB）。需 Kaggle API token。
- **KAIST** — Kaggle `adlteam/kaist-dataset`。需 Kaggle API token。

Kaggle API token 設定（FLIR/KAIST 必須）：
1. kaggle.com → 頭像 → Settings → API → **Create New API Token**，下載 `kaggle.json`。
2. `mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   或設環境變數 `KAGGLE_USERNAME` / `KAGGLE_KEY`。

> 注意：專案根目錄的 `LLVIP/` 是官方 **GitHub repo**（文件/工具），**不是**資料集。真資料集由 `download.py --llvip` 下載到 `data/raw/LLVIP`。

---

## 資料前處理

1. 將各數據集下載至 `data/raw/`（見上方「資料下載」）。
2. 轉換成 YOLO 格式（每張圖一個 `.txt`，內容 `class cx cy w h` 已正規化）。只保留 `person` 類別 → 重新對應為 class id `0`。

```bash
python scripts/convert_llvip.py --src data/raw/LLVIP --dst data/yolo
python scripts/convert_flir.py  --src data/raw/FLIR  --dst data/yolo
python scripts/convert_kaist.py --src data/raw/KAIST --dst data/yolo
# KAIST 想納入 cyclist:
#   python scripts/convert_kaist.py --src data/raw/KAIST --dst data/yolo --classes person cyclist

# 轉換後驗證格式與計數
python scripts/verify_dataset.py --dst data/yolo
```

轉換器行為：
- 只保留 `person`，重新映射為 class id `0`。
- 影像預設 **symlink** 進 `data/yolo`（省空間）；加 `--copy` 改為複製。
- 依檔名雜湊做確定性 train/val/test 切分（預設 0.8/0.1/0.1），可用 `--val`/`--test` 調整；多次執行結果一致。

各數據集注意事項：
- **LLVIP** — 標註為 PASCAL VOC XML，只有 `person`。使用 IR 影像。
- **FLIR ADAS** — COCO 格式 JSON，過濾 category `person`（丟掉 `car`/`bicycle`）。
- **KAIST** — 自訂 `.txt` 標註，類別 `person`、`people`、`cyclist`。自行決定哪些對應到 `person`（常見做法：保留 `person`，選擇性納入 `cyclist`）。依需求略過 `ignore`/遮擋標記。

`configs/ir_person.yaml`：

```yaml
path: data/yolo
train: images/train
val: images/val
test: images/test
names:
  0: person
```

---

## 訓練

包裝腳本（已內建 IR 增強策略）：

```bash
python scripts/train.py                     # 預設 100 epochs
python scripts/train.py --epochs 150 --batch 32 --device 0
python scripts/train.py --resume runs/ir_person_yolo11n/weights/last.pt
```

`train.py` 已針對熱影像設定：`hsv_h=0 hsv_s=0`（關色彩）、`hsv_v=0.2`（保留亮度抖動）、保留幾何增強與 mosaic、最後 10 epoch 關 mosaic、`patience=30` early stop。

或直接用 Ultralytics CLI：

```bash
yolo detect train model=yolo11n.pt data=configs/ir_person.yaml \
  epochs=100 imgsz=640 batch=16 project=runs name=ir_person_yolo11n
```

IR 訓練小技巧：
- 從 COCO 預訓練的 `yolo11n.pt` 開始（遷移學習優於從頭訓練）。
- 關閉/減弱色彩增強（HSV）— 熱影像沒有有意義的色彩；保留幾何增強（翻轉、縮放、mosaic）。
- 謹慎混合數據集：FLIR 為單一熱影像，LLVIP/KAIST 需抽取 IR 串流。統一解析度 / 長寬比。

---

## 評估

```bash
yolo detect val \
  model=runs/ir_person_yolo11n/weights/best.pt \
  data=configs/ir_person.yaml \
  split=test
```

輸出 `person` 類別的 mAP@0.5、mAP@0.5:0.95、precision、recall。

---

## 推論

```bash
yolo detect predict \
  model=runs/ir_person_yolo11n/weights/best.pt \
  source=path/to/ir_image_or_folder \
  imgsz=640 conf=0.25
```

```python
from ultralytics import YOLO
model = YOLO("runs/ir_person_yolo11n/weights/best.pt")
results = model("path/to/thermal.jpg")
results[0].show()
```

---

## 匯出

```bash
# ONNX
yolo export model=runs/ir_person_yolo11n/weights/best.pt format=onnx
# TensorRT（邊緣 / Jetson）
yolo export model=runs/ir_person_yolo11n/weights/best.pt format=engine
```

---

## 開發計畫

- [x] 數據集轉換腳本（LLVIP / FLIR / KAIST）。
- [x] 統一的 train/val/test 切分（確定性雜湊切分）。
- [x] 訓練 / 驗證 / 一鍵前處理腳本。
- [x] `yolo11n` baseline 訓練 + 指標表（見下）。

### Baseline 結果（LLVIP + FLIR，100 epoch, RTX 4060 Ti）

資料：25716 圖 / 70583 person box（train 20625 / val 2514 / test 2577）。KAIST 跳過（Kaggle 版無標註）。

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| val  | 0.957 | 0.641 | 0.928 | 0.903 |
| test | 0.949 | 0.615 | 0.932 | 0.881 |

權重：`runs/detect/runs/ir_person_yolo11n/weights/best.pt`（yolo11n, 2.58M params）。
- [ ] Domain-mix 消融實驗（單一數據集 vs. 混合）。
- [ ] 邊緣部署效能測試（ONNX / TensorRT 延遲）。

---

## 授權

本 repo 程式碼：MIT（或自行選擇）。
數據集維持其**原始授權** — 見各數據集來源。本 repo 不重新散布數據集影像或標註。
