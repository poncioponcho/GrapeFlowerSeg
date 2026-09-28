# 官方赛题契约存档

赛题：**阳光玫瑰葡萄花穗完整性判断与分割挑战赛**（2026 讯飞 AI 开发者大赛）

本文件是官方规则的存档。权威来源是随数据下发的
`README_数据说明.md`、`B榜必读！！！提交说明.md`，以及冻结的官方脚本
`src/eval/official_oracle/official_evaluate.py`、`src/submit/official/submission.py`。
**以下任何一条与官网不一致时，以官网为准并同步修改本文件与 `configs/default.yaml`。**

---

## 0. 赛题身份（2026-09-26 已核对官方赛题页）

官方地址：`https://challenge.xfyun.cn/topic/info?type=GrapeFlowerSeg&option=ssgy`

| 项 | 值 |
|---|---|
| 举办方 | 中国农业大学 |
| 参赛团队 | 393 |
| 奖金 | 一等奖 5000 元、二等奖 3000 元、三等奖 2000 元（各 1 名，共 3 名） |
| 赛程 | 9/3–9/28；A 榜 9/3–9/22；**B 榜 9/24 0:00–9/28 17:00** |
| 现场答辩 | 前三名受邀总决赛，作品成绩 70% + 答辩 30% |
| 数据采集 | TCL RayNeo X2 AR 智能眼镜，第一人称视角 |

⚠️ **本项目即为此赛题。** `data/raw/` 下载的数据与该赛题页完全对应
（702 训练 / 94 A / 106 B、类别映射、Mask mAP、`b_submission.zip` 契约）。
原先收到的任务书描述的是另一个赛题（果粒着色三分类），见 `docs/recon_findings.md`。

### 0.1 官方页面内部矛盾（已核对，以冻结脚本为准）

赛题页 §三.3) 与 §四.3)A) 对 B 榜包内容的描述**互相冲突**：

| 出处 | 说法 |
|---|---|
| §三.3)（第 3 条） | `b_submission.zip` 含 `result.json` + `solution_commit.txt` |
| §四.3)A) | `b_submission.zip` 根目录含 **`submit.zip`** + `solution_commit.txt` |
| `B榜必读！！！提交说明.md` | `result.json` + `solution_commit.txt` |
| **冻结的 `submission.py::read_submission`** | 只接受 `result.json` + `solution_commit.txt`，否则直接报错 |

**结论：以冻结的官方校验器为准**，即 `result.json` + `solution_commit.txt`。
§四 里的 `submit.zip` 应是从其他赛题复制的笔误 —— 本项目已用官方
`validate_b_submission.py` 实测通过，确认这一解读正确。

### 0.2 赛后复核范围的两处说法

| 出处 | 范围 |
|---|---|
| 赛题页 §三.3)4) | 排行榜 **Top3**（或获奖序列） |
| `B榜必读！！！提交说明.md` | 排行榜 **Top10** |

两者冲突。按**更严格**的口径准备：即使只进 Top10 也要备好可复现的 `solution.zip`。

---

## 1. 任务与类别

对单张田间葡萄图像，检测并分割每个花穗，判断其完整性。

| 提交 `category_id` | 含义 | 训练 COCO `category_id` |
|---|---|---|
| `0` | 完整花穗 | `1`（name `"0"`，supercategory `flower_cluster`） |
| `1` | 不完整花穗 | `2`（name `"1"`，supercategory `flower_cluster`） |

⚠️ **训练编号与提交编号不同，必须转换。** 本项目只在
`src/data/coco.py::gt_category_to_submit` 一处转换。

不接受第三类别；`ignore_regions` **不是**类别，不参与预测输出。

## 2. 数据

| 项 | 值 |
|---|---|
| 训练 | 702 图 / 351 源图 / 1602 普通实例 / 1822 忽略区域 |
| 训练结构 | 每张源图各带一张 `_aug1` 翻转增强图（351 × 2 = 702） |
| A 榜测试 | 94 图（无标注） |
| B 榜测试 | 106 图 = 53 原图 + 53 张 `_aug1`，**全部参与评测** |
| 图像尺寸 | 2048 × 1152（`size` 按 `[height, width]` = `[1152, 2048]`） |
| 标注格式 | COCO（`instances_train.json`、`ignore_regions_train.json`） |

忽略区域位于 `ignore_regions` 键下（非 `annotations`），无 `category_id`，
自带 `images` 列表。

## 3. 评测

**主指标：COCO Mask mAP@[0.50:0.95]**

- IoU 阈值：`[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]`
- 101 点插值，`area=all`，`maxDets=[1, 10, 100]`（**固定，调用方不可修改**）
- 对两个类别与十个 IoU 阈值取平均，范围 `[0,1]`，越高越好

### 忽略区域规则（类别无关）

1. 每个 IoU 阈值下先匹配同类别普通真实实例。
2. 对未匹配的预测，计算 `IgnoreOverlap = area(pred ∩ union(ignore)) / area(pred)`。
3. `IgnoreOverlap >= 0.50` → 该预测在该阈值下**不计 TP，也不计 FP**。
4. 忽略区域不计入 GT 召回率/AP 分母，也不是第三类。
5. 同一忽略区域对两个预测类别都生效。

### 评分命令

```bash
python src/eval/official_oracle/official_evaluate.py \
  --gt_json  <GT>  --ignore_json <IGNORE> \
  --submission_json <result.json> \
  --output_json <score.json> --error_json <error.json>
```

成功时 `score.json` 的 `evaluation.score` 即最终得分。本项目通过
`src/eval/mask_map.py::run_official_scorer` 以子进程调用，**从不重新实现该指标**。

## 4. 提交契约

### 4.1 `b_submission.zip` 根目录

**恰好两个普通文件**，不得嵌套目录、目录条目、重复文件、符号链接或加密文件：

```text
b_submission.zip
├── result.json
└── solution_commit.txt
```

**不要**放入说明、README、图像信息、校验工具、评分脚本或 `solution.zip`。

### 4.2 `result.json`

- UTF-8；顶层**恰好** `{"version": "1.0", "results": [...]}`
- `results` 覆盖本次全部 106 张图，每张**恰好一条**记录，不得重复、遗漏或增加外部图像
- 每条记录**恰好** `{"image_id", "instances"}`
- `image_id` 是**完整文件名含 `.jpg`**，与公开清单完全一致
- 无预测时 `instances` 写 `[]`，**不能删除整条记录**

每个非空实例**恰好**三个字段：

| 字段 | 要求 |
|---|---|
| `category_id` | 整数 `0`（完整花穗）或 `1`（不完整花穗） |
| `score` | `[0,1]` 内有限数值；不能是布尔值 / NaN / Infinity |
| `segmentation` | **必须且只能**含 `size` 与 `counts` 的 COCO compressed RLE |

- `size` 固定按 `[height, width]` 顺序，两项必须为整数
- 必须先把预测掩码**恢复到原图分辨率**再编码为真实 COCO compressed RLE
- `counts` 必须是压缩 RLE 字符串，不接受未压缩整数数组
- 不接受 Polygon、多边形坐标、二值图片路径或 Base64
- 同图不同花穗分别输出实例，不能合并成一个掩码
- JSON 所有层级不允许重复字段
- 任一必填字段缺失、类型错误、额外字段或非法实例 → **整次提交无效**（不是丢弃该条）

### 4.3 `solution_commit.txt`

UTF-8，`key=value`，**必须且只能**含以下五个字段，每个恰好出现一次：

```text
solution_name=solution.zip
hash_algorithm=SHA-256
solution_sha256=<64位十六进制SHA-256>
solution_size=<solution.zip的正整数字节数>
b_data_version=B-v1
```

- 不加注释、空行、字段两侧多余空格或额外字段
- `solution_size` 无前导零、不带单位/小数点/千位分隔符
- SHA 计算对象固定为**完整可独立运行的原始 `solution.zip`**；不能对单个权重、
  解压目录、`result.json` 或 `b_submission.zip` 求哈希
- 重新压缩会改变 SHA，因此**打包后必须保留原文件，不得重压**

### 4.4 体积上限

| 对象 | 上限 |
|---|---|
| `b_submission.zip` | 1 GiB（1,073,741,824 字节） |
| `result.json`（解压后） | 1 GiB |
| `solution_commit.txt` | 65,536 字节 |
| 赛后提交的 `solution.zip` | 压缩包 ≤ 5 GB，解压后 ≤ 10 GB |

## 5. 提交次数与最终成绩

- B 榜每队**总共 3 次**提交，不因日期重置，期间不展示实时分数或排名
- 每次提交独立对应一套「原始模型包—预测结果—SHA 声明」
- 平台按服务器接收时间确定截止前**最后一次有效提交**，**不取历史最高分**
- 最后一次若格式无效，不覆盖更早的有效记录
- 截止后不能补写或替换最终 SHA

## 6. 赛后复核

成绩确认后，Top10 队伍提交最终有效声明对应的原始 `solution.zip`。赛事方核对
SHA-256 与字节数、检查包内容、**断网运行**、比对预测并复算分数。
**复核 Mask mAP 与榜单成绩的绝对差原则上不得超过 0.005。**

模型包结构要求：

```text
solution.zip
├── model/                 # 模型权重
├── config/                # 配置
├── src/                   # 模型及推理源码
├── inference.py           # 统一推理入口
├── requirements.txt       # 依赖及版本
└── README.md              # 安装、推理和复现说明
```

统一推理入口：

```bash
python inference.py \
  --input_dir test_images/ \
  --weights model/best.pt \
  --output_path reproduced_result.json
```

复现环境：Ubuntu 22.04 / Python 3.10 / PyTorch 2.2 / CUDA 12.1 / 单张 NVIDIA GPU /
Batch Size 1。集成、TTA、多阶段推理与自定义后处理必须完整提交并说明。

## 7. 禁止事项

- ❌ 外部数据、额外标注数据
- ❌ 测试集查表、硬编码答案、人工制作预测、读取隐藏标注
- ❌ 测试图像及预测结果用于训练、伪标签或人工标注
- ❌ 在线 API 调用
- ✅ 允许公开可下载、可离线运行的预训练权重与开源模型，但须在模型包 README 中
  列明名称、版本与下载地址

## 8. 关键时间

| 事件 | 时间 |
|---|---|
| B 榜开放 | 2026-09-24 00:00 |
| **B 榜截止** | **2026-09-28 17:00** |
