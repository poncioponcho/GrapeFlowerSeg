# OOF report (grouped K-fold by source image)

> ⚠️ **SYNTHETIC DATA** - these numbers come from generated data used
> to self-test the pipeline. They are **not** competition results.

- folds: **5**
- n: **1631**
- **Macro-F1 (OOF): `0.79955`**
- accuracy: 0.79522
- MAE (ordinal): 0.20785

## Confusion matrix (rows = truth, cols = prediction)

| truth \ pred | 0 | 1 | 2 | support |
|---|---|---|---|---|
| **0** | 365 | 103 | 3 | 471 |
| **1** | 72 | 565 | 76 | 713 |
| **2** | 2 | 78 | 367 | 447 |

## Adjacent-class confusion (the real error budget)

- 0 -> 1: 103   |  1 -> 0: 72
- 1 -> 2: 76   |  2 -> 1: 78
- distant (0 <-> 2): 5
- total errors: 334 of 1631
- **share of errors that are adjacent: 0.985**

## Per class

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| 0 | 0.8314 | 0.7749 | 0.8022 | 471 |
| 1 | 0.7574 | 0.7924 | 0.7745 | 713 |
| 2 | 0.8229 | 0.8210 | 0.8219 | 447 |

## Per fold

| fold | n | Macro-F1 | accuracy |
|---|---|---|---|
| 0 | 326 | 0.78402 | 0.80982 |
| 1 | 326 | 0.80817 | 0.80368 |
| 2 | 326 | 0.78658 | 0.79141 |
| 3 | 326 | 0.77659 | 0.76994 |
| 4 | 327 | 0.79911 | 0.80122 |
