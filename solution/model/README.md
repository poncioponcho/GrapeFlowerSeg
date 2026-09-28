# model/

模型权重目录。

```text
model/
├── model.py     # PanicleSegmenter：自包含网络定义与推理（.load() / .predict()）
└── fold{0..4}.pt  # 各折权重（torchvision Mask R-CNN，variant 在权重内）
```

`inference.py` 通过 `from model.model import PanicleSegmenter` 导入：

```python
class PanicleSegmenter:
    @classmethod
    def load(cls, weights_path_or_list, *, device="cuda",
             short_side=None) -> "PanicleSegmenter": ...
    def eval(self) -> None: ...
    def predict(self, image_path, width, height) -> list[dict]:
        """返回 [{"category_id": 0|1, "score": float,
                  "segmentation": {"size": [h, w], "counts": str}}, ...]"""
```

`load()` 接受单个路径或路径列表；列表即多折集成（按类别 NMS）。

**推理分辨率**：`short_side=None` 表示沿用权重内记录的**训练**分辨率（640）；
传入整数或 `"640,1024"` 可指定一个或多个尺度，每个（权重 × 尺度）组合的检测
都并入同一个并集再做按类别 NMS。`inference.py` 默认传
`DEFAULT_INFER_SHORT_SIDES = (640, 1024)`：同权重在留出折上测过，
单用 1024 与单用 640 相当（均值 +0.18pp，噪声量级），而**两个尺度融合**
稳定更优（均值 +0.52pp，5 折中 4 折上涨）。复现榜单成绩必须使用该默认值。

替换权重后必须**重新打包 `solution.zip` 并重新生成 `solution_commit.txt`**，
SHA 必须与本次提交的结果一一对应。
