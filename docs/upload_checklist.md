# 上传检查清单 — 防止交错比赛

B 榜**累计只有 3 次提交**，交错一次就少一次，且最后一次有效提交才是最终成绩。
本清单必须在每次上传前后各过一遍。

---

## 0. 本赛题身份（上传前逐字对照平台页面）

| 项 | 值 |
|---|---|
| **competition id** | `GrapeFlowerSeg` |
| 赛题名 | 阳光玫瑰葡萄花穗完整性判断与分割挑战赛 |
| 官方页 | https://challenge.xfyun.cn/topic/info?type=GrapeFlowerSeg&option=ssgy |
| 举办方 | 中国农业大学 |
| 上传文件 | `b_submission.zip`（**只有一个文件**） |
| 包内根目录 | 恰好 `result.json` + `solution_commit.txt` |

> 平台页面的 URL 里必须出现 **`type=GrapeFlowerSeg`**。如果看到的是别的 `type=`，**停手**。

---

## 1. 上传前：机器已自动检查（`build_submission.py` 会打印）

构建时会打印身份横幅并跑 6 项预检，任何一项 FAIL 都会**拒绝构建**：

```
[OK ] competition id configured: GrapeFlowerSeg
[OK ] test manifest exists
[OK ] testB image count: manifest has 106, config expects 106
[OK ] image dimensions: all 106 are 2048x1152
[OK ] augmented pairing (53 + 53): 53 originals + 53 _aug1 copies
[OK ] test images present on disk: all 106 found
```

**"augmented pairing (53 + 53)" 是本赛题最强的身份特征** —— B 榜是 53 张原图 +
53 张 `_aug1` 翻转图。如果这个数字不是 53/53，说明数据不是本赛题的。

---

## 2. 上传前：人工确认（3 项，30 秒）

- [ ] 平台页面 URL 含 `type=GrapeFlowerSeg`
- [ ] 平台页面标题是「阳光玫瑰葡萄花穗完整性判断与分割挑战赛」
- [ ] 我上传的文件是 `b_submission.zip`（不是 `submit.zip`、不是 `result.json`、
      不是 `solution.zip`）

---

## 3. 不要上传到这些赛题（工作区里同时存在）

| id | 赛题 | 为什么容易搞混 |
|---|---|---|
| `824` | 妮娜皇后葡萄果粒着色阶段分级 | 三分类 / Macro-F1 / 提交 `predictions.csv`，格式完全不同 |
| `825` | 复杂田间环境下妮娜皇后葡萄果粒实例分割挑战赛 | **同为实例分割，最危险**；类别定义与图像清单不同 |
| `Spark-X2.5` | Spark-X2.5 端侧模型创新挑战赛 | 完全不同，工作区另有目录 |

`824` 和本赛题的提交格式差异大，交错了一般会被格式校验挡下。
**`825` 才是真正的风险点** —— 它同样是 COCO 实例分割、同样可能用 `result.json`，
格式校验不一定能救你。所以第 2 节的人工确认不能省。

---

## 4. 不要放进包里的东西

包内根目录**只能**有两个文件。以下任何一样都会导致整次提交无效：

- ❌ `SUBMISSION_IDENTITY.md`（本项目的身份记录，**放在包外**）
- ❌ `solution.zip`（赛后复核才交，不进比赛阶段提交包）
- ❌ `README`、说明文档、校验工具、评分脚本
- ❌ 目录条目、符号链接、加密文件、重复文件名

`build_submission.py` 与 `verify_submission.py` 已对以上全部做硬拦截。

---

## 5. 上传后：立即确认

- [ ] 平台显示「提交成功」，且提交记录归属本赛题
- [ ] 提交编号已记下（`outputs/submissions/<tag>/SUBMISSION_IDENTITY.md` 里也记一份）
- [ ] 平台显示的分数合理（空包为 0；真实模型应为正数）
- [ ] 保存本次对应的**原始 `solution.zip`**，与 `solution_commit.txt` 一起归档

> ⚠️ 平台按**服务器接收时间**确定截止前最后一次有效提交，**不取历史最高分**。
> 所以最后一次上传必须是最想留下的那一版。

---

## 6. 归档要求（赛后复核用）

- 每次提交单独存一套：`solution.zip` + `result.json` + `solution_commit.txt` +
  `SUBMISSION_IDENTITY.md`，按提交编号或时间命名
- **绝不重新压缩 `solution.zip`**：声明绑定的是原始字节的 SHA-256，
  重新打包即使内容相同也会改变 SHA，导致复核不通过
- 赛后 Top3（赛题页口径；B榜必读说明书写 Top10）需提交最终有效提交对应的原始包

---

## 7. 万一交错了

1. **立刻停止继续提交**（还剩的次数更宝贵了）
2. 若交错的是无效提交，平台不会覆盖更早的有效记录 —— 先确认它是否被记为「有效」
3. 用剩余次数重新提交正确版本，并在提交前重跑本清单第 1、2 节
4. 若已无剩余次数，联系主办方（AICompetition@iflytek.com）说明情况
