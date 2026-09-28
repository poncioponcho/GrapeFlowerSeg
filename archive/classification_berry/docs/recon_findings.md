# Recon findings — verified facts vs. unverified claims

Date: 2026-09-26 · Author: engineering recon pass before P0

This file records what was **independently verified**, what could **not** be
verified, and every place where the task brief disagrees with a public source.
The brief said: *"如果比赛规则与我上面列出的有出入，以官网赛题页为准并立即指出差异。"*
Those differences are listed here first.

---

## 1. Discrepancies that need your attention

### 1.1 ⚠️ Deadline conflict (highest impact)

| Source | Claimed date |
|---|---|
| Task brief | **B 榜截止 2026-09-28 17:00** |
| Tavondo competition page (last checked 2026-09-04, indexed 2026-09-25) | 报名截止 **2026-09-28**; 提交截止 **2026-10-24**; 赛事结束 2026-10-24 |
| AI赛事备忘录 (indexed ~2026-09-22) | 报名截止 **2026-09-27** |
| Official page `challenge.xfyun.cn/topic/info?type=824` | **login-gated** — content could not be read |

**Assessment.** Every public source puts the *registration* deadline at
9/27–9/28 and the *submission* deadline at **2026-10-24**. The brief's
"B 榜截止 9/28 17:00" looks like the **registration** deadline being read as the
submission deadline.

**Why it matters.** If the real submission deadline is 10/24, the whole plan
changes: there is roughly **four weeks** rather than two days, which turns
"minimum viable baseline" into "proper hyper-parameter search + full ensemble".
If it really is 9/28, the plan in the brief is right.

**Action required.** Someone with an account must open the official page and
confirm. Nothing downstream should be re-planned until then. I did **not**
change the working assumption — the pipeline is built to the tighter deadline so
it is safe either way.

### 1.2 ⚠️ The evaluation metric is not confirmed

The brief states Macro-F1:
`f1_score(y_true, y_pred, labels=[0,1,2], average="macro", zero_division=0)`.

Tavondo's page states plainly: *"官方 metric 还没有从当前抓取结果中确认，仍需人工核对"*
(the official metric has not been confirmed from the current scrape). The
official page is login-gated, so it could not be checked directly.

**Consequence.** The implementation is written to that exact definition, and the
scoring layer is isolated in `src/eval/macro_f1.py` behind a differential test.
If the official metric turns out to be, say, micro-F1 or weighted-F1, exactly one
function changes and the test suite tells you what else breaks.

### 1.3 ⚠️ The dataset is not on this machine

`data/raw/` is **empty**. There is no `labels.csv`, no images, and no
`sample_submission.csv` anywhere under the workspace. The official download is
behind the same login.

**Consequence.** P0 is complete and fully tested, but **P1–P3 are blocked**:
no training, no OOF score, no real submission. See §4.

### 1.4 Note: there is a *second*, different grape competition

`challenge.xfyun.cn/topic/info?type=825` is **复杂田间环境下妮娜皇后葡萄果粒实例分割挑战赛**
(instance segmentation of grape berries). Same host, same fruit, different task
and different metric. Worth double-checking that the account is registered for
**type=824** (classification) and not type=825 (segmentation).

### 1.5 Confirmed: no external data

Tavondo reproduces the official wording: *"外部葡萄、果粒或果实数据不能用于训练、微调或
类别校准"* — external grape/berry/fruit data may not be used for training,
fine-tuning, or class calibration. This matches the brief's constraint, and
`configs/default.yaml::competition.external_grape_data_allowed` is `false`.

---

## 2. Verified: the historical repository is real and reusable

`https://github.com/poncioponcho/lane-detection-challenge` — **HTTP 200**,
README 17,102 bytes, 248 tracked paths, default branch `main`, shallow clone
succeeds. Verified directly with `curl`/`git`, not inferred.

It is the team's HardLane entry (科大讯飞 2026 AI 开发者大赛, lane detection in
adverse weather). What was actually reused:

| Reference asset | Reused as | Notes |
|---|---|---|
| `src/common/checksum.py` | `src/common/checksum.py` | SHA-256 helpers, near-verbatim |
| `src/common/io_utils.py` | `src/common/io_utils.py` | adapted to CSV |
| `src/data/split_by_clip.py` | `src/data/split_by_source.py` | **clip → source image** grouping; the leakage-assertion idea is the key carry-over |
| `src/submit/{prepare,pack,verify}_submit.py` | `src/submit/*` | retargeted `.lines.txt` tree → single `predictions.csv` |
| `configs/default.yaml` | `configs/default.yaml` | single-source-of-truth discipline |
| README 经验教训 §1, §4 | `docs/experiments.md`, `src/eval/macro_f1.py` | "offline conclusions don't transfer"; "a custom metric needs a differential test" |

One deliberate **inversion**: HardLane's `submit.zip` required a `submit/` wrapper
directory. This competition requires the root to contain `predictions.csv`
directly. `pack_flat` in `src/submit/pack_submit.py` therefore *rejects* any
arcname containing a separator.

Their lane-detection model layer (CLRNet, geometry post-processing) was correctly
not reused — different task.

---

## 3. What the brief got right

Verified or consistent:

- Task is single-berry 3-class classification, ordinal labels, adjacent classes
  hard to separate — consistent with the official description.
- Split is by original source image, so local validation must be grouped. This is
  the single most important modelling constraint and the pipeline enforces it.
- Colour is the label signal, so aggressive HSV augmentation is harmful and
  MixUp/CutMix must be off. Encoded as `forbidden_augmentations`.
- Image sizes are non-uniform with grey padding outside the mask; needs
  resize/pad/normalise. Handled in the transform pipeline.
- B-board submission wraps `submit.zip` plus `solution_commit.txt`.
- 3-submission limit on the B board.

**Unverified arithmetic note:** 1769 + 222 + 218 = 2209, which is internally
consistent, but none of those four numbers could be confirmed against the
official page. They are marked `[UNVERIFIED]` in `configs/default.yaml` and are
used only for synthetic-data sizing, never for a reported score.

---

## 4. What is blocked, and on what

| Stage | Status | Blocker |
|---|---|---|
| P0 engineering skeleton | ✅ complete, 116 tests green | — |
| P1 baseline training | ⛔ code complete, never executed | no dataset; no torch/timm |
| P2 TTA / ensemble / error analysis | ⛔ code complete, never executed | needs P1 |
| P3 B-board submission | ⛔ pipeline proven on synthetic data only | needs P1/P2 |

To unblock: download the official training set into `data/raw/`, then run
`make manifest && make split`. The source-grouping report will immediately reveal
whether the file-name → source-image convention guess is correct (it must find
~130 groups, not one per image).

---

## 5. Reproduction of this recon

```bash
curl -sS -o /dev/null -w '%{http_code}\n' -L \
  https://github.com/poncioponcho/lane-detection-challenge          # 200
git clone --depth 1 https://github.com/poncioponcho/lane-detection-challenge /tmp/lane-ref
curl -sS -L 'https://challenge.xfyun.cn/topic/info?type=824' | wc -c   # SPA shell, no body
```

Third-party aggregator pages consulted: `tavondo.com/zh/competitions/xfyun-824`,
`competehub.dev/zh/competitions/xfyun824`. **Aggregators are not authoritative** —
they are used only to identify what to check on the official page.
