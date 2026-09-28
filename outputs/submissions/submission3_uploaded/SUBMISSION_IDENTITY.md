# Submission identity record

> This file lives **next to** the archive, never inside it. The platform
> accepts exactly `result.json` + `solution_commit.txt` in the root and
> rejects any extra file.

- generated: `2026-09-28T11:46:41+08:00`
- competition id: **`GrapeFlowerSeg`**
- competition name: 阳光玫瑰葡萄花穗完整性判断与分割挑战赛
- official page: https://challenge.xfyun.cn/topic/info?type=GrapeFlowerSeg&option=ssgy
- split: `testB`
- upload this file: `b_submission.zip`
- images: 106   instances: 683

## Declared solution package

- `solution_sha256`: `753747a9df77a8b32ea5581b3e66f320f20e9697bc3ec595a5fc8ee3a96a56a3`
- `solution_size`: `1633587670`
- archive sha256: `a49c1b04a36a9e33c782005e69b296d63e3f4d7a8f500919e16089c2ddb81f00`

## Pre-flight checks

| check | result | detail |
|---|---|---|
| competition id configured | PASS | GrapeFlowerSeg |
| test manifest exists | PASS | /Users/seyonmacbook/WorkBuddy AI/妮娜皇后葡萄果粒着色阶段分级/data/manifests/images_test_b.json |
| testB image count | PASS | manifest has 106, config expects 106 |
| image dimensions | PASS | all 106 are 2048x1152 |
| augmented pairing (53 + 53) | PASS | 53 originals + 53 _aug1 copies |
| test images present on disk | PASS | all 106 found under images |

## Do not upload this to

- **824** 妮娜皇后葡萄果粒着色阶段分级 — 三分类 / Macro-F1 / 提交 predictions.csv —— 格式与本赛题完全不同
- **825** 复杂田间环境下妮娜皇后葡萄果粒实例分割挑战赛 — 同为实例分割，最易混淆；但类别与图像清单不同
- **Spark-X2.5** Spark-X2.5 端侧模型创新挑战赛 — 完全不同的赛题，工作区另有目录

## Reminder

- The B board allows **3 submissions total**; a mis-uploaded file burns one.
- After a successful upload, confirm the platform shows the expected
  competition and that the submission is recorded as valid.
- Do **not** re-zip `solution.zip` afterwards: the declaration is bound to
  the original bytes.
