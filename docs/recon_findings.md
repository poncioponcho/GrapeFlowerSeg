# 侦察结论 — 任务书与真实数据的差异

日期：2026-09-26 · 在开工前对赛题、历史仓库、本机数据做的一次核对

本文件记录**已独立核验的事实**、**无法核验的项**，以及**任务书与实际数据不一致的每一处**。
任务书原话：*"如果比赛规则与我上面列出的有出入，以官网赛题页为准并立即指出差异。"*

---

## 0. 赛题身份已确认

用户随后给出了目标赛题页
`https://challenge.xfyun.cn/topic/info?type=GrapeFlowerSeg&option=ssgy`。

**该赛题就是本机 `data/raw/` 里数据所属的赛题** ——
「阳光玫瑰葡萄花穗完整性判断与分割挑战赛」，官方页与数据逐项对应：

| 核对项 | 官方赛题页 | 本机数据 | 一致 |
|---|---|---|---|
| 训练集 | 702 张 | 702 张 | ✅ |
| A 榜 | 94 张 | 94 张 | ✅ |
| B 榜 | 106 张 | 106 张 | ✅ |
| 训练类别映射 | `1`=完整花穗、`2`=不完整花穗 | 同 | ✅ |
| 提交类别映射 | `0`=完整花穗、`1`=不完整花穗 | 同 | ✅ |
| 指标 | Mask mAP@[0.50:0.95] | 官方脚本一致 | ✅ |
| B 榜提交 | `b_submission.zip` | 官方校验器一致 | ✅ |
| B 榜窗口 | 9/24 0:00–9/28 17:00 | 同 | ✅ |

**结论：无需更换数据，本项目已在该赛题上。** 原任务书描述的「果粒着色三分类」
是另一个赛题，不在此次目标内。

---

## 1. ⚠️ 最关键：任务书描述的是另一个赛题

### 1.1 实际下载到的数据是什么

`data/raw/` 内 6 个文件（`训练集A.zip` 579 MB、`测试集A.zip`、`B榜测试集B.zip`、
`提交样例A.zip`、`B榜提交样例.zip`、`B榜必读！！！提交说明.md`）。解压后：

| 项 | 实际值 | 来源 |
|---|---|---|
| 赛题 | **阳光玫瑰葡萄花穗完整性判断与分割挑战赛** | 官方 `official_evaluate.py` 头部注释 |
| 任务 | **实例分割**（COCO） | `README_数据说明.md` |
| 类别 | 完整花穗 / 不完整花穗（2 类） | 同上 |
| 训练 | 702 图、1602 实例、1822 忽略区域 | 同上，已用脚本复核一致 |
| 训练类别编号 | COCO `1`=完整花穗、`2`=不完整花穗 | 同上 |
| **提交类别编号** | **`0`=完整花穗、`1`=不完整花穗** | `B榜必读！！！提交说明.md` |
| 图像尺寸 | 2048×1152 | `images_test_b.json` |
| A 榜 | 94 图 | `README_数据说明.md` |
| B 榜 | 106 图 = 53 原图 + 53 张 `_aug1` | `B榜必读！！！提交说明.md` |
| 指标 | **COCO Mask mAP@[0.50:0.95]** | 官方 `official_evaluate.py` |
| 提交 | `b_submission.zip` → 根目录恰好 `result.json` + `solution_commit.txt` | `submission.py` |

### 1.2 任务书写的是什么

| 任务书 | 实际数据 |
|---|---|
| 果粒着色三分类（0=着色初期 / 1=着色中期 / 2=充分着色） | 花穗实例分割，2 类 |
| 评测指标 **Macro-F1** | **COCO Mask mAP@[0.50:0.95]** |
| 提交 `predictions.csv`（`image_id,label`） | 提交 `result.json`（COCO compressed RLE） |
| `submit.zip` 根目录含 `predictions.csv` | `b_submission.zip` 根目录含 `result.json` + `solution_commit.txt` |
| 2209 张 / 1769 训练 / 222 A / 218 B | 702 训练 / 94 A / 106 B |
| 按原始源图分组划分 | 成立，但源图是「花穗照片」，且每张源图**各带一张 `_aug1` 翻转增强图** |
| 未提及模型包 | **必须声明可运行 `solution.zip` 的真实 SHA-256 与字节数** |
| 未提及赛后复核 | 赛后 Top10 断网复现，Mask mAP 绝对差 ≤ 0.005 |

**结论：这是两个不同的赛题。** 本项目按**真实数据**重建。若本意是参加三分类赛题，
说明数据下错了赛题，需要重新下载。

### 1.3 任务书里那些「未核验」的数字

任务书中的 2209 / 168 / 1769 / 130 / 222 / 218 全部**无法在真实数据中复核**，
且与实际的 702 / 94 / 106 不符。已全部从配置中移除，替换为实测值。

---

## 2. 已核验：历史仓库真实且工程层可复用

`https://github.com/poncioponcho/lane-detection-challenge` — HTTP 200，
README 17,102 字节，248 个路径，浅克隆成功。直接 `curl`/`git` 验证，非推测。

复用的部分：

| 历史仓库 | 本项目 | 说明 |
|---|---|---|
| `src/common/checksum.py` | `src/common/checksum.py` | SHA-256（本赛题变成强制项） |
| `src/common/io_utils.py` | `src/common/io_utils.py` | 略作改造 |
| `src/data/split_by_clip.py` | `src/data/split_by_source.py` | clip → 源图分组；**泄漏断言**是核心传承 |
| `src/submit/{prepare,pack,verify}_submit.py` | `src/submit/*` | 三段式安全网，改判为 result.json 契约 |
| `configs/default.yaml` 纪律 | 同 | 常量唯一事实源 |
| README 经验教训 §4「自造尺子必须差分验证」 | `src/eval/mask_map.py` + 官方脚本冻结 | 评分口径不可自证 |

一处**刻意的反向**：历史仓库要求 `submit/` 外层目录，本赛题要求根目录扁平，
因此 `pack_b_submission` 拒绝任何带路径分隔符的成员名。

模型方法层（CLRNet、车道线几何后处理）与本赛题无关，未复用。

---

## 3. 本机环境与数据可用性

| 项 | 状态 |
|---|---|
| 训练数据 | ✅ 已解压 `data/train/`（702 图 + 2 个标注 json） |
| 测试数据 | ✅ `data/testA/`（94 图）、`data/testB/`（106 图） |
| 公开图像清单 | ✅ `data/manifests/images_test_b.json`（只有 id/宽高，无答案） |
| 官方评分脚本 | ✅ 已冻结 `src/eval/official_oracle/official_evaluate.py` |
| 官方校验器 | ✅ 已冻结 `src/submit/official/validate_b_submission.py` |
| 提交链路 | ✅ 已端到端验证，保底提交官方校验 PASS |
| 模型 | ⏳ 训练中 |

---

## 4. 阻塞与风险

| 项 | 状态 | 说明 |
|---|---|---|
| 提交链路 | ✅ 无阻塞 | 保底提交已就位，即使模型失败也有有效提交 |
| 模型训练 | ⏳ 时间风险 | 本机为 Apple Silicon，无 CUDA；2048×1152 原图需降分辨率训练 |
| 官方评测器依赖 | ⚠️ 版本差异 | 官方声明 `numpy==1.26.4`，本机 `2.1.3`；实测官方脚本可运行，若出现差异需回退复现 |
| 赛后复核 | ⚠️ 需保留原包 | `solution.zip` 打包后**不可重压**，SHA 与预测必须一一对应 |

---

## 5. 本次侦察的复现命令

```bash
curl -sS -o /dev/null -w '%{http_code}\n' -L \
  https://github.com/poncioponcho/lane-detection-challenge     # 200
git clone --depth 1 https://github.com/poncioponcho/lane-detection-challenge /tmp/lane-ref

# 数据事实全部来自本机解压后的官方文件，可用以下命令复核：
python -m data.coco            # 702 图 / 351 源图 / 1602 实例 / 1822 忽略区域
python -m data.split_by_source # 分组 5 折 + 泄漏断言
```

第三方聚合站（tavondo、competehub）仅用于**定位要核对什么**，不作为权威来源。
