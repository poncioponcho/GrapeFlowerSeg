# 大文件清单（不进 git，走 GitHub Release 附件）

这 9 个文件合计 5.5 GB，**无法放进仓库**：GitHub 对仓库内单文件有 **100 MiB 硬上限**，
private 仓库同样适用（官方文档 *About large files on GitHub*）。这不是隐私决定 ——
比赛数据本就是官方公开的，官方规则 §7「禁止事项」也没有禁止传播数据的条文。

替代方案是 **Release 附件**：官方文档 *About releases* 明确「每个附件须小于 2 GiB、
单次 release 最多 1000 个附件、无总量与带宽限制」。最大的是 1.5 GB 的 `solution.zip`，
全部装得下。附件不进 git 历史，所以仓库体积不受影响；private 仓库的 release 同样私有。

## 获取方式

仓库 → **Releases** → tag `artifacts-v1` → 下载所需附件。

`outputs/release_assets/`（已 gitignore）里放了这 9 个文件的**硬链接**，名字已改为唯一，
可直接拖拽上传。硬链接不额外占磁盘。

## 清单

| 附件 | 字节数 | 说明 |
|---|---|---|
| `训练集A.zip` | 579,412,343 | 官方训练集原始压缩包 |
| `fold0.pt` | 351,088,433 | 5 折权重（fold 0）—— **提交 #3 实际使用的权重** |
| `fold1.pt` | 351,088,433 | 5 折权重（fold 1） |
| `fold2.pt` | 351,088,433 | 5 折权重（fold 2） |
| `fold3.pt` | 351,088,433 | 5 折权重（fold 3） |
| `fold4.pt` | 351,088,433 | 5 折权重（fold 4） |
| `fold0_brokenmask.pt` | 351,088,433 | ⚠️ 2026-09-26 掩膜头损坏的废 checkpoint，**仅作留痕，不要用于推理** |
| `submission2_solution.zip` | 1,633,666,575 | 提交 #2 的 `solution.zip`（平台分 0.31903） |
| `submission3_solution.zip` | 1,633,587,670 | **提交 #3 的 `solution.zip`（平台分 0.32924，最终成绩）** |

> 两个 `solution.zip` 被改名，因为 release 附件名必须唯一。**内容未做任何改动** ——
> 声明绑定的是原始字节的 SHA-256，改名不影响校验。提交 #3 的
> `solution_commit.txt` 里写的是 `solution_name=solution.zip`，指的就是
> `submission3_solution.zip`。

## SHA-256

```text
fede2caa93cc0df67a15f2c7a20318a74f7650fcb86c7239f75a89bf3980ac3a  训练集A.zip
ed91633ea9ae0e2a7dcdabd8741329a3bc3ca3b891a9ba6fb487f022e07cfba1  fold0.pt
459da5fe3c8861bf15823470fde1b9d36144a886e32ef2547cd89697ff56ec7e  fold1.pt
eb6a5ad3d7e7dbd32e8af07501a9cd98188b227275c618ee2a825a885a88e283  fold2.pt
6daa0938986b0d6eb6e7fd1f02d246dfbbc860d35848447d1781ae05eef63b60  fold3.pt
1609973fa48911d520c1cad1d3765c4abefdebad176d786aa00f52474f239ccc  fold4.pt
3d37a0b2db5347dc73f60ae614def9cad1c487d8b33a15354d9c9761bf55a200  fold0_brokenmask.pt
a731ccf1c12f72619eb6665b052a5ccbf9482630d7c8ba5511d604424d8c2f85  submission2_solution.zip
753747a9df77a8b32ea5581b3e66f320f20e9697bc3ec595a5fc8ee3a96a56a3  submission3_solution.zip
```

后两个与各自的 `solution_commit.txt` 声明一致 —— 即赛后复核要交的原始包。

## 校验

```bash
shasum -a 256 <下载的文件>
# 与上表比对；submission3_solution.zip 必须等于 753747a9df77a8b32ea5581b3e66f320f20e9697bc3ec595a5fc8ee3a96a56a3
```

## 为什么不用 Git LFS

Git LFS 免费额度为 1 GB 存储，而这 9 个文件需要 5.5 GB，需额外购买数据包；
Release 附件免费且额度足够。分卷压缩进 git 也可行，但会把仓库从 894 MB 撑到 1.4 GB 以上，
拖慢每次 clone，不划算。
