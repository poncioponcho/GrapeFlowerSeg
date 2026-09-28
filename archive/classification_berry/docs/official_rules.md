# Official rules archive — BerryColorGrade (type=824)

P0 task 3 required archiving the official scoring statement so a later metric
change cannot go unnoticed. **Caveat up front:** the official page is
login-gated, so what is archived here is the third-party reproduction plus the
team's own reading. The authoritative text must be pasted in below by someone
with an account — the placeholder block is marked `TODO`.

---

## 1. Competition identity

| Field | Value | Source |
|---|---|---|
| Name | 果粒着色阶段智能分级挑战赛 (BerryColorGrade) | official page title |
| Platform | 2026 iFLYTEK AI 开发者大赛 | official |
| Official URL | https://challenge.xfyun.cn/topic/info?type=824 | official |
| Track | 计算机视觉 / 算法赛 | aggregator |
| Sponsor | 中国农业大学 | aggregator |
| Teams entered | 604 | aggregator snapshot |

## 2. Official description (as reproduced by the aggregator)

> 妮娜皇后葡萄是高端红色鲜食葡萄品种，其商品外观和市场价值与果皮着色程度密切相关。
> 随着果实发育，果皮由绿色、黄绿色逐渐转为粉红色、红色和深红色。准确识别果粒着色阶段，
> 可为田间监测、果穗管理、分批采收和商品分级提供依据。
> 传统判断主要依赖人工观察，容易受到经验、光照、拍摄角度和个体差异影响。
> 与此同时，阴影、高光反射、设备色差、果粉覆盖、果粒遮挡及着色不均等因素……

Stated difficulty focus: fine-grained separation of **adjacent** colour stages,
not detection or segmentation. Target berries may be partially occluded, and are
affected by shadow, specular highlight, device colour cast, bloom (果粉) and
uneven colouring.

## 3. Class definitions (ordinal)

| Label | Name | Chinese |
|---|---|---|
| 0 | `early_veraison` | 着色初期 |
| 1 | `mid_veraison` | 着色中期 |
| 2 | `full_coloration` | 充分着色 |

Labels are ordinal (0 < 1 < 2). The confusable pairs are 0↔1 and 1↔2.
Mirrored in `configs/default.yaml::competition.classes`.

## 4. Scoring statement

### 4.1 The metric as implemented

The leaderboard ranks by **Macro-F1**, equivalent to:

```python
from sklearn.metrics import f1_score
f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
```

Implemented once, in `src/eval/macro_f1.py::macro_f1`, with a second,
independent pure-numpy implementation (`macro_f1_reference`) used as a
differential oracle. `tests/test_macro_f1.py` asserts both agree with
scikit-learn to 1e-15 across many random draws.

### 4.2 Verification status

> ⚠️ **NOT CONFIRMED against the official page.** The aggregator states the
> official metric has not been confirmed from its scrape. Treat the definition
> above as the working assumption, not as verified fact.

`Accuracy` and `MAE` are reported for display only and do not affect ranking.

### 4.3 TODO — paste the official text here

```
TODO(owner-with-account): open https://challenge.xfyun.cn/topic/info?type=824,
switch to the 赛题数据 / 评分标准 section, and paste the verbatim scoring
description here. Then update configs/default.yaml::competition.metric if it
differs, and run `make test` — the differential test will flag any mismatch.
```

## 5. Data

| Item | Value | Status |
|---|---|---|
| Total images | 2209 | unverified |
| Source images (果穗图) | 168 | unverified |
| Public train | 1769 images / 130 source images | unverified |
| A-board test | 222 | unverified |
| B-board test | 218 | unverified |

Train format: image files + `labels.csv` with fields `image_id, file_name, label`.
Crop sizes are non-uniform; areas outside the berry mask are grey-filled.
Split is **by original source image** — a berry crop appears in exactly one subset.

Arithmetic check: 1769 + 222 + 218 = 2209 ✓ (internally consistent, still unverified).

## 6. Submission contract

### A-board — `submit.zip`
- archive **root** contains **exactly one** file: `predictions.csv`
- `predictions.csv`: UTF-8, first line exactly `image_id,label`
- every test `image_id` appears **exactly once**
- `label ∈ {0, 1, 2}`
- missing / duplicated / extra id, or an illegal label ⇒ **invalid submission**

### B-board — `b_submission.zip`
- archive root contains exactly `submit.zip` and `solution_commit.txt`
- the embedded `submit.zip` must independently satisfy the A-board contract
- `solution_commit.txt` binds the submission to a solution version

### Limits
- B-board cumulative submissions: **3**

Enforced by `src/submit/verify_submit.py`, with one named test per failure class
in `tests/test_submit_verify.py`.

## 7. Constraints

- ❌ No external grape / berry / fruit data for training, fine-tuning, or class
  calibration.
- ✅ Pretrained weights limited to official ImageNet pretrained models.
- ⚠️ All pretrained-weight use must be recorded.

## 8. Deadlines — see `docs/recon_findings.md` §1.1

| Event | Date | Confidence |
|---|---|---|
| Registration closes | 2026-09-27 / 09-28 | medium |
| Submission closes | **2026-10-24** | medium — **conflicts with the brief's 2026-09-28 17:00** |
| Competition ends | 2026-10-24 | medium |

Do not re-plan around either date until the official page is read.
