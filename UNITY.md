# Unity 整合 — best.onnx (yolo11n IR person)

ONNX 給 **Unity Sentis**（`com.unity.sentis`，Barracuda 後繼）使用。

## 模型規格

| 項目 | 值 |
|---|---|
| 檔案 | `runs/detect/runs/ir_person_yolo11n/weights/best.onnx`（10 MB, opset 15） |
| Input | `images` `float32[1,3,640,640]`，**RGB**、值域 **0–1（/255）**、無 mean/std、NCHW |
| Output | `output0` `float32[1,5,8400]` |
| Output 5 維 | `[cx, cy, w, h, person_score]`，bbox 為 **640×640 像素座標** |
| 8400 | anchor 數（80²+40²+20²），單類別 person |
| NMS | **未內嵌**，需在 C# 端做 |

前處理需 **letterbox**（保持長寬比縮到 640，補灰邊 114），否則框會偏。

## 安裝

Unity Package Manager → Add package by name → `com.unity.sentis`（用 2.x）。
把 `best.onnx` 拖進 `Assets/`（Unity 會產生 `ModelAsset`）。

## C# 範例（Sentis 2.x）

```csharp
using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;

public class IRPersonDetector : MonoBehaviour
{
    public ModelAsset modelAsset;          // 拖 best.onnx
    const int SIZE = 640;
    const float CONF = 0.35f;              // 信心閾值
    const float IOU  = 0.45f;              // NMS IoU

    Worker worker;

    void Start()
    {
        var model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);   // 或 CPU
    }

    public List<Box> Detect(Texture2D img)
    {
        // 1) letterbox 到 640x640，寫入 tensor (1,3,640,640) RGB /255
        var (input, scale, padX, padY) = Preprocess(img);

        // 2) 推論
        worker.Schedule(input);
        var output = worker.PeekOutput() as Tensor<float>;
        var data = output.DownloadToArray();   // 長度 5*8400
        input.Dispose();

        // 3) decode + NMS。output layout: [5, 8400]（channel-major）
        //    data[c*8400 + i] = 第 i 個 anchor 的第 c 維
        const int N = 8400;
        var cand = new List<Box>();
        for (int i = 0; i < N; i++)
        {
            float score = data[4 * N + i];
            if (score < CONF) continue;
            float cx = data[0 * N + i], cy = data[1 * N + i];
            float w  = data[2 * N + i], h  = data[3 * N + i];
            // 640 座標 → 去 letterbox → 原圖座標
            float x0 = (cx - w / 2 - padX) / scale;
            float y0 = (cy - h / 2 - padY) / scale;
            cand.Add(new Box { x = x0, y = y0, w = w / scale, h = h / scale, score = score });
        }
        return NMS(cand, IOU);
    }

    // ---- letterbox 前處理 ----
    (Tensor<float>, float, float, float) Preprocess(Texture2D img)
    {
        int iw = img.width, ih = img.height;
        float scale = Mathf.Min((float)SIZE / iw, (float)SIZE / ih);
        int nw = Mathf.RoundToInt(iw * scale), nh = Mathf.RoundToInt(ih * scale);
        float padX = (SIZE - nw) / 2f, padY = (SIZE - nh) / 2f;

        var buf = new float[3 * SIZE * SIZE];
        // 灰底 114/255
        for (int k = 0; k < buf.Length; k++) buf[k] = 114f / 255f;
        var px = img.GetPixels32();
        for (int y = 0; y < nh; y++)
        for (int x = 0; x < nw; x++)
        {
            int sx = Mathf.Min(iw - 1, (int)(x / scale));
            int sy = Mathf.Min(ih - 1, (int)(y / scale));
            var c = px[(ih - 1 - sy) * iw + sx];   // Texture2D y 反向
            int dx = x + (int)padX, dy = y + (int)padY;
            int idx = dy * SIZE + dx;
            buf[0 * SIZE * SIZE + idx] = c.r / 255f;
            buf[1 * SIZE * SIZE + idx] = c.g / 255f;
            buf[2 * SIZE * SIZE + idx] = c.b / 255f;
        }
        var t = new Tensor<float>(new TensorShape(1, 3, SIZE, SIZE), buf);
        return (t, scale, padX, padY);
    }

    // ---- NMS ----
    List<Box> NMS(List<Box> boxes, float iouTh)
    {
        boxes.Sort((a, b) => b.score.CompareTo(a.score));
        var keep = new List<Box>();
        foreach (var b in boxes)
        {
            bool ok = true;
            foreach (var k in keep)
                if (IoU(b, k) > iouTh) { ok = false; break; }
            if (ok) keep.Add(b);
        }
        return keep;
    }

    float IoU(Box a, Box b)
    {
        float x1 = Mathf.Max(a.x, b.x), y1 = Mathf.Max(a.y, b.y);
        float x2 = Mathf.Min(a.x + a.w, b.x + b.w), y2 = Mathf.Min(a.y + a.h, b.y + b.h);
        float inter = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
        return inter / (a.w * a.h + b.w * b.h - inter + 1e-6f);
    }

    void OnDestroy() => worker?.Dispose();

    public struct Box { public float x, y, w, h, score; }
}
```

## 注意

- **RGB 順序**：ultralytics 用 RGB。Unity `Color32` 也是 RGBA，上面直接取 `.r/.g/.b` 正確。
- **IR 影像**：熱影像多為單通道，Unity 載入後若是灰階，R=G=B 即可（模型 3 通道，灰階複製 3 份等效）。
- **Sentis API 版本**：上面是 2.x（`Worker` / `Schedule` / `Tensor<float>`）。Sentis 1.x 為 `WorkerFactory.CreateWorker` / `TensorFloat` / `worker.Execute`，需微調。
- output 是 **channel-major** `[5,8400]`：第 c 維第 i 個 anchor = `data[c*8400 + i]`。別搞成 `[8400,5]`。
- 想要更省事可改 `nms=True` 重匯出（內嵌 NMS），但 Sentis 對 NMS 算子支援有限，多數情況 C# 端做較穩。
