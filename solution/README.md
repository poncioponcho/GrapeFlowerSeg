# solution.zip - 阳光玫瑰葡萄花穗完整性判断与分割

复现环境（题面规定）：Ubuntu 22.04 / Python 3.10 / PyTorch 2.2 / CUDA 12.1 /
单张 NVIDIA GPU / Batch Size = 1。

## 安装

```bash
pip install -r requirements.txt
```

## 推理

```bash
python inference.py \
  --input_dir test_images/ \
  --weights model/fold0.pt model/fold1.pt model/fold2.pt model/fold3.pt model/fold4.pt \
  --output_path reproduced_result.json
```

`--weights` 接受一个或多个权重：多权重时按"检测并集 + 按类别 NMS(IoU 0.5) + 每图
截断 100 个检测"集成，与提交的 `result.json` 生成逻辑一致。

**推理分辨率**：默认 `--short_side 640,1024`（见 `model/model.py` 的
`DEFAULT_INFER_SHORT_SIDES`），与本次提交的 `result.json` 生成时一致 ——
5 折 × 2 个尺度 = 10 组检测做按类别 NMS 融合。权重内记录的是**训练**分辨率 640
（当时的时间预算决定），单独沿用会让掩膜边界偏糙。**复现榜单成绩请使用默认值，
不要传 `--short_side`。**

可选：`--manifest images_test_b.json` 提供逐图宽高；`--score_threshold` 控制输出
阈值（默认 0.05，与提交包一致）；`--device cuda|cpu`。

输出为 `result.json` 格式：`category_id` 0=完整花穗、1=不完整花穗；
`segmentation` 为 COCO compressed RLE。每张图恰好一条记录，无检测时
`"instances": []`。

## 重要

- 赛后复核会断网运行本包，并比对复现结果与榜单成绩（Mask mAP 绝对差 ≤ 0.005）。
- 打包后**不要**重新压缩或修改 `solution.zip`：声明绑定的是原始字节的 SHA-256。
- 使用的预训练权重与开源模型必须在下表中列明名称、版本与下载地址。
- 禁止联网调用 API、禁止使用外部数据或测试集查表。

## 依赖与预训练权重

| 项目 | 名称 | 版本 | 来源 |
|---|---|---|---|
| 框架 | PyTorch | 2.2.0 | https://pytorch.org |
| 框架 | torchvision | 0.17.0 | https://pytorch.org |
| 预训练 | maskrcnn_resnet50_fpn（COCO，仅训练初始化用；推理包不联网下载） | torchvision 0.17 内置 | https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth |

## 目录结构

```text
solution.zip
├── model/
│   ├── model.py            # PanicleSegmenter（自包含，含 .load() / .predict()）
│   └── fold{0..4}.pt       # 各折权重（variant 记录在权重内，推理分辨率见 model.py）
├── inference.py            # 统一推理入口
├── requirements.txt        # 依赖及版本
└── README.md               # 本文件
```
