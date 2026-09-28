# 训练集数据说明

本压缩包只包含训练集，不包含测试集A或任何测试答案。

## 目录结构

```text
训练集/
├── train/
│   └── images/                         # 702张训练图像
├── annotations/
│   ├── instances_train.json            # COCO实例分割普通标注
│   └── ignore_regions_train.json       # 类别无关忽略区域
└── README_数据说明.md
```

## 类别说明

- 训练COCO标注：`category_id=1`、类别名`0`表示完整花穗。
- 训练COCO标注：`category_id=2`、类别名`1`表示不完整花穗。
- `ignore_regions_train.json`中的区域不是第三个类别，可在训练或本地验证时忽略。
- 正式提交文件采用另一套编号：`category_id=0`表示完整花穗，`category_id=1`表示不完整花穗。

训练集共有702张图像、1602个普通花穗实例和1822个忽略区域。
