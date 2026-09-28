# 项目长期记忆 — 妮娜皇后葡萄果粒着色阶段分级

> 目录名来自任务书（824 三分类），但**实际做的是 825/花穗实例分割赛题**（详见 README
> 与 `docs/recon_findings.md`）。最终平台分 **0.32924**（B 榜 2026-09-28 17:00 截止）。

## 仓库与发布边界

- 公开仓库：`git@github.com:poncioponcho/GrapeFlowerSeg.git`（PUBLIC），只含 **78 文件 /
  0.46MB** 的代码与文档。**云端不是本项目的备份**（本地另有 6.4GB 未上传）。
- 永不入库：`data/`（主办方条款）、`outputs/`（权重 351MB/个、solution.zip 1.6GB）、
  `src/eval/official_oracle/` + `src/submit/official/`（**主办方第三方代码**）、
  `.workbuddy-ai/`（内部笔记）。
- 缺官方脚本时 `tests/test_official_freeze.py` 应 **skip 而非 fail**（已实现），
  保证别人 clone 后 `make test` 全绿。

## 发布事故与规则（已发生，务必记住）

- ⚠️ **`.gitignore` 里排除目录必须加前导斜杠。** 首次发布用了裸 `data/`，git 会匹配任意
  层级的同名目录 → `src/data/` 被整体排除，`coco.py` 与 `split_by_source.py`
  **没进仓库**，clone 下来 import 直接失败。已改锚定为 `/data/`、`/outputs/`
  （提交 `fb0f72a`）。**发布前用 `git status --ignored` 复核被排除清单**，
  不要只看 `git ls-files`。
- 仓库曾被设为 PUBLIC 约 2 小时（18:03–18:3x），后改 private。**改 private 不能撤回
  已公开的历史**（可能被缓存/fork）。
- **GitHub 单文件 100MB 是硬限制，private 仓库同样拒收**。本地有 9 个文件超限：
  `data/raw/训练集A.zip`(553M)、6 个 checkpoint(335M/个)、2 个 solution.zip(1.5G/个)。
  → **推荐方案：GitHub Release 附件**（官方文档：每个附件 <2GiB、单 release 最多 1000 个、
  无总量与带宽限制），9 个文件全部可挂上去，不进 git 历史、不撑大仓库、不需要 LFS、免费。
  private 仓库的 release 同样私有。备选：分卷压缩进 git（会把仓库撑到 1.4GB+，不推荐）、
  Git LFS（免费额度 1GB，不够 5.6GB，要买数据包）、外部硬盘。
- 关于数据能否公开：查过 `docs/official_rules.md` §7「禁止事项」，**只禁止外部数据、
  测试集查表/硬编码、读取隐藏标注、用测试图训练、在线 API**，**没有禁止传播数据集**的条文。
  所以训练集/测试集不是"不能传"，纯粹是 100MB 技术上限挡住的。
- private ≠ secret：GitHub 员工可见；日后若转 public，整个历史都会公开。不要放凭据。

## 仓库最终范围（用户决定：private 后"除凭据外尽量全传"）

- **已入库 1217 文件 / 894MB**：全部代码文档 + `data/`（902 张图、清单、折划分、样例提交）
  + 主办方官方脚本 + `outputs/`（报告/日志/预测/三次提交记录）+ `.workbuddy-ai/`
- **仍排除（纯体积，非隐私）**：`data/raw/训练集A.zip`(553M)、
  `outputs/checkpoints/`(6×335M)、`outputs/submissions/submission*/solution.zip`(2×1.5G)
  —— GitHub 单文件 100MB 硬限制，private 同样适用；需外部备份或 Git LFS
- 推送前必须验证：`git rev-list --objects --all | git cat-file --batch-check` 里
  **可达 blob 无 >100MB**（否则 push 被拒）。实测可达 875MB，最大 80MB。

## 又两个 .gitignore 陷阱（实际踩过，README 已记）

1. **不支持行内注释** —— 行尾 `# 553 MB` 会成为模式的一部分，规则静默失效，
   结果 9 个大文件差点被一起提交（`git add` 因此被宿主杀掉，还留下 5.7GB 的
   `.git` 与 `index.lock`）。注释必须单独成行。
2. 被 SIGTERM 杀掉的 `git add` 会留下 **`.git/index.lock`**（0 字节），
   后续 git 命令报 `Unable to create index.lock`。确认无 git 进程后直接删掉即可。
   `git gc --prune=now` 可回收被杀进程留下的未引用对象。

## 复用地图（下次同类比赛直接照这个换）

**可直接复用（约 45 文件，未绑定赛题）**
- `src/common/`：配置加载、SHA-256、IO
- `src/submit/`：`result_json` / `solution_commit` / `pack_submission` / `verify_submission`
  四段式（提交契约的通用骨架）
- `src/data/split_by_source.py`：**按源图分组 K 折 + 泄漏硬断言** —— 本仓库最值钱的资产
- `scripts/`：`train_segmentation` / `predict_test` / `evaluate_local` / `repro_check` /
  `night_runner`（带门槛的夜间串行任务队列）/ `threshold_scan` / `verify_training_pipeline`
- `tests/`、`Makefile`、`pytest.ini`

**必须替换（约 15 文件，赛题绑定）**
- `configs/default.yaml`（21 处赛题常量：competition id、类别映射、图像数、身份特征）
- `src/submit/identity.py`（赛题专属上传前预检 —— **防交错比赛的关键，但复用时要改身份特征**）
- `src/data/coco.py`（类别重编号）、`src/eval/mask_map.py` + 冻结官方脚本
- `solution/`（模型与推理）、`README.md`、`docs/` 四篇

**做法**：另建 `ml-competition-template` 仓库只放通用层，新赛题 clone 后填赛题层；
不要在原仓库上改（历史会混入两个赛题的文档，容易误传）。

## 硬约定（血的教训，别丢）

1. **任何改动的结论必须来自 5 折配对比较。** 单折 140 图 mAP 噪声 ±0.7pp ——
   E9 就是只看 fold-4 的 +0.88pp 而误判（5 折均值仅 +0.18pp）。
2. **"跑满轮数 + loss 正常" ≠ 在训练。** 检测手段：对 checkpoint 权重取哈希
   （`torch.load` → 逐张量 float32 bytes → sha256）比对。E4 因续训 lr=0 空转 2.9h。
3. **复现检查按"指标相关项严格、浮点项带容差"设计**：category 严格、掩膜按 IoU≥0.995、
   score 容差 1e-5。跨设备（MPS/CUDA）逐字节相等是不现实的目标。
4. **平台取"最晚"的有效提交，不取最高分** → 末位改动有下行风险，未验证别交。
5. **官方评分脚本会被宿主机删除保护拦下**：`safe_unlink(score.json)` 在文件已存在时
   触发保护 → 评分进程 exit 1。**用全新不存在的输出路径**即可绕开。
6. `build_submission.py` 清理暂存权重用 **rename**（`retire_staged`）而非 `unlink`，
   否则删除保护会中断构建。
7. 含中文引号的 git 提交消息一律用 heredoc（`git commit -F -`）。
