# 阳光玫瑰葡萄花穗完整性判断与分割

2026 讯飞 AI 开发者大赛 · 花穗实例分割（B 榜 2026-09-24 ~ 2026-09-28 17:00，累计 3 次提交）。
**最终平台分 0.32924**（COCO Mask mAP@[0.50:0.95]）。

单张田间图像 → 检测并分割每个葡萄花穗，二分类：

| 提交 `category_id` | 含义 | 训练 COCO `category_id` |
|---|---|---|
| `0` | 完整花穗 | `1` |
| `1` | 不完整花穗 | `2` |

主指标 **COCO Mask mAP@[0.50:0.95]**（101 点插值，`area=all`，`maxDets=100`，
类别无关 ignore 区域参与评测）。

---

## ⚠️ 先读这一段

**任务书与真实数据是两个不同赛题。** 收到的任务书写的是「妮娜皇后葡萄果粒着色三分类
（Macro-F1 / predictions.csv）」，但 `data/raw/` 里实际下载的是**花穗实例分割**
（COCO Mask mAP / result.json）。本项目已按**真实数据**重建，差异逐条记录在
`docs/recon_findings.md`。若你本意是参加三分类赛题，说明数据下错了赛题。

**B 榜提交必须附带 `solution_commit.txt`**，声明原始 `solution.zip` 的真实 SHA-256
与字节数。缺声明 / SHA 格式错 / 与模型包不一致 → 该次提交无效。赛后 Top3（赛题页口径；
B榜必读说明书写 Top10）会断网复现并比对分数（Mask mAP 绝对差 ≤ 0.005）。
这条任务书里完全没提，但它是硬性的。

---

## 🚨 别交错比赛

工作区里同时存在多个赛题，其中 **`825` 同样是 COCO 实例分割**，是最危险的一个。
B 榜**累计只有 3 次提交**，交错一次就少一次。

**本赛题身份（上传前必须对照平台页面）：**

| 项 | 值 |
|---|---|
| competition id | **`GrapeFlowerSeg`** |
| 赛题名 | 阳光玫瑰葡萄花穗完整性判断与分割挑战赛 |
| 官方页 | https://challenge.xfyun.cn/topic/info?type=GrapeFlowerSeg&option=ssgy |

**三重防护，都已落地：**

1. **构建时硬预检** —— `build_submission.py` 在打包前会跑 6 项身份检查，
   任何一项失败就**拒绝构建**：

   ```bash
   make preflight     # 单独跑，失败时退出码非 0，可接在 && 前面
   ```

   检查项：competition id 已配置 / 测试清单存在 / 图像数 = 106 /
   尺寸全为 2048×1152 / **`_aug1` 配对为 53+53** / 106 张图在磁盘上都存在。
   其中 **53+53 的增强配对是本赛题最强的身份特征**。

2. **构建时打印身份横幅** —— 列出 competition id、官方 URL，以及**它不是什么**
   （824 / 825 / Spark-X2.5 各自为什么不同）。

3. **身份记录旁挂** —— 每次构建都在包**外**写一份 `SUBMISSION_IDENTITY.md`，
   记录 competition id、包内声明、archive 的 SHA-256、预检结果与上传提醒。
   平台禁止包内出现第三个文件，所以它只能放在旁边。

**上传前人工确认 3 件事**（30 秒）：URL 含 `type=GrapeFlowerSeg`、页面标题正确、
上传的是 `b_submission.zip`。完整清单见 **`docs/upload_checklist.md`**。

---

## 状态

| 阶段 | 状态 |
|---|---|
| 提交链路（result.json + SHA 声明 + 打包 + 官方校验） | ✅ 完成并已验证 |
| 数据层 / 分组划分 / 冻结官方评分 | ✅ 完成，**116 个测试全绿** |
| 模型训练 | ✅ 5 折（light resnet50_fpn v1 @ short_side 640，12 epoch，MPS） |
| 真实 B 榜提交 | ✅ 3 次提交完成，**最终平台分 0.32924** |

平台分进展：单折 fold-0 **0.31382** → 5 折集成 **0.31903** → 5 折 × 2 尺度集成 **0.32924**。

留出折上的证据（按源图分组、冻结官方评分脚本）：

| fold | 0 | 1 | 2 | 3 | 4 | 均值 |
|---|---|---|---|---|---|---|
| 单模型 @640 | 0.3434 | 0.3226 | 0.3500 | 0.3453 | 0.3602 | 0.3443 |
| 单模型 640+1024 融合 | 0.3490 | 0.3283 | 0.3572 | 0.3441 | 0.3691 | 0.3495 |

**任何改动的结论必须来自 5 折配对比较。** 单折 140 图的 mAP 噪声在 ±0.7 pp 量级：
推理分辨率单用 1024 时在 fold-4 上看着涨了 +0.88 pp，5 折均值只有 +0.18 pp
（标准差 0.69 pp），与 0 无法区分 —— 这个改动因此被否掉，而"两个尺度融合"
（均值 +0.52 pp，5 折中 4 折上涨）被采纳。详见 `docs/experiments.md`。

---

## 快速开始

```bash
make test                # 116 个测试，约 2 秒，无 GPU 无网络
make empty-submission    # 生成保底提交并通过官方校验器
```

**保底提交**：`outputs/submissions/b_submission.zip`，106 张图、0 预测、格式合法、
官方校验器 PASS。这是刻意的安全网 —— 即使模型训练失败，截止前也一定有一次有效提交。

---

## 提交链路（本项目最可靠的部分）

```bash
make predict            # 对 testB 推理 -> outputs/predictions/raw_predictions_testB.json
make submission         # 打包 solution.zip -> 生成 SHA 声明 -> 生成 result.json
                        # -> 打包 b_submission.zip -> 官方校验器验证
make verify             # 不改动任何文件，只重新校验现有提交包
```

`b_submission.zip` 根目录**恰好**两个普通文件：

```
b_submission.zip
├── result.json
└── solution_commit.txt
```

`result.json` 顶层恰好 `{"version": "1.0", "results": [...]}`；每张测试图恰好一条记录
（无检测写 `"instances": []`，**不能删记录**）；每个实例恰好
`{category_id, score, segmentation}`，其中 `segmentation` 是 COCO compressed RLE。

`solution_commit.txt` 恰好 5 个字段、UTF-8、无注释无空行：

```
solution_name=solution.zip
hash_algorithm=SHA-256
solution_sha256=<64 位十六进制>
solution_size=<正整数字节数>
b_data_version=B-v1
```

### 三次提交建议

1. **第 1 次**：先用保底包（或单折模型）验证整条链路与平台接收
2. **第 2 次**：完整多折模型 + TTA
3. **第 3 次**：留作修正

---

## 训练与本地验证

```bash
make data               # 数据统计（含源图分组报告）
make split              # 按源图分组 5 折（断言同源图不跨折）
make train              # Mask R-CNN，逐折保存 checkpoint
make evaluate           # 用【冻结的官方评分脚本】在留出折上算 Mask mAP
```

**关键约束：`IMG_x.jpg` 与 `IMG_x_aug1.jpg` 是同一张源图。** 本地验证必须把它们
放在同一折，否则 Mask mAP 虚高、选模型全错。这条由硬断言强制
（`src/data/split_by_source.py`），不是靠约定。

测试集没有标注，所以**唯一可信的本地依据**就是分组留出折上的官方脚本分数。
不要用「按图随机划分」的任何数字做决策。

---

## 目录结构

```
configs/default.yaml          常量唯一事实源（代码里不得硬编码）
src/common/                   配置加载、SHA-256、IO
src/data/
  coco.py                     COCO 读取 + 类别重编号 + 源图分组
  split_by_source.py          分组 K 折 + 泄漏断言
src/eval/
  mask_map.py                 调用冻结的官方评分脚本
  official_oracle/            ⛔ 官方评分脚本原件（逐字节冻结，SHA 守护）
                              —— 第三方代码，未纳入版本控制，见下方说明
src/submit/
  result_json.py              result.json 构造与逐字段校验
  solution_commit.py          5 字段 SHA 声明
  pack_submission.py          打包（根目录扁平约束）
  verify_submission.py        本地检查 + 调用官方校验器
  official/                   ⛔ 官方校验器/解析器原件（冻结，同上）
solution/                     可运行模型包源码（打包进 solution.zip）
tests/                        116 个测试
archive/classification_berry/ 三分类赛题的旧代码（已归档，不再使用）
```

### 未纳入版本控制的内容

| 路径 | 原因 |
|---|---|
| `data/` | 比赛数据，主办方条款不允许再分发；用 `make data` / `make split` 重建清单与折 |
| `outputs/`、`solution/model/*.pt` | 权重 351 MB/个、`solution.zip` ~1.6 GB，超 GitHub 单文件限制 |
| `src/eval/official_oracle/`、`src/submit/official/` | **主办方自己的评分/校验脚本**（第三方代码），需从官方提交包取回；`src/official_freeze.json` 记录了它们应有的 SHA-256 |

缺官方脚本时 `tests/test_official_freeze.py` 会 **skip 而不是 fail**，
所以 clone 下来直接 `make test` 也是绿的（2 passed / 7 skipped）。

### 设计要点

- **官方脚本冻结。** 官方评分与校验脚本逐字节复制进 `src/`，SHA-256 记录在
  `src/official_freeze.json`，`tests/test_official_freeze.py` 在文件被改动时红灯。
  评分口径不可被悄悄改动。
- **类别重编号只有一处。** 训练 `1/2` → 提交 `0/1` 只在 `src/data/coco.py` 完成，
  写错会把两个类对调。
- **提交包先本地验，再上传。** `verify_submission.py` 复刻官方读取器的全部规则
  （根目录恰好两文件、无目录项、无符号链接、无加密、成员非空），并额外调用官方
  校验器作为最终裁决。

### 已知限制

- **赛后复核**：Top10 需提交最终有效提交对应的原始 `solution.zip`，断网复现并与榜单
  比对（Mask mAP 绝对差 ≤ 0.005）。本机训练用 Apple MPS，官方复现环境为 CUDA，
  两者浮点路径不同：实测同权重在 CPU 上重跑，59 个实例里有 1 个掩膜 IoU 为 0.9985
  （0.15% 像素差异），score 差 2e-06 —— 远在 0.005 容差内，但复现脚本必须按
  "指标相关项严格、浮点项带容差"来设计，不能要求逐字节相等。
- 官方评测器声明依赖 `numpy==1.26.4`；本机为 `2.1.3`，实测官方脚本可运行，
  若出现差异需回退复现。
- 推理分辨率默认 `(640, 1024)`（见 `solution/model/model.py`），**必须与提交时一致**，
  否则赛后复现算不出同样分数。

---

## 文档

| 文件 | 内容 |
|---|---|
| `docs/recon_findings.md` | 任务书与真实数据的逐条差异、已核验事实 |
| `docs/official_rules.md` | 官方赛题契约存档（类别、指标、提交格式、约束） |
| `docs/experiments.md` | 实验日志、环境 pin、评分代码变更记录、缺陷留痕 |
| `docs/upload_checklist.md` | 上传前后逐项清单（防交错比赛、归档要求） |
