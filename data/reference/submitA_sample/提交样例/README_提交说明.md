# 提交样例说明

## 正式上传文件

正式比赛时请直接上传UTF-8编码的`result.json`，不要上传本说明文件、`tools/`目录、依赖文件、示例文件或整个`提交样例.zip`。

本压缩包中的`result.json`覆盖测试集A全部94张图像，`instances`均为空，是可以直接上传并通过格式校验的空预测样例，预期得分为`0.0`。参赛者应在此结构上填入自己的预测实例。

## result.json结构

- 顶层必须且只能包含`version`和`results`。
- `version`固定为字符串`"1.0"`。
- `results`必须覆盖测试集A全部94张图像；每个图像文件名恰好出现一次，不能遗漏、重复或加入测试集外图像。
- 每个结果项必须且只能包含`image_id`和`instances`。
- `image_id`填写包含扩展名的完整图像文件名。
- 未检测到花穗时，`instances`必须写成空数组`[]`。
- 每个预测实例必须且只能包含`category_id`、`score`和`segmentation`。
- `category_id=0`表示完整花穗，`category_id=1`表示不完整花穗。
- `score`必须是0～1之间的有限数值。
- `segmentation`必须是COCO compressed RLE对象，且只能包含`size`和`counts`。
- `size`顺序固定为`[height, width]`，必须与原图尺寸一致。
- `counts`必须是可解码的非空字符串；空掩码、polygon、多边形坐标、图片路径或Base64图片均不接受。

主指标为COCO实例分割`Mask mAP@[0.50:0.95]`，分数越高排名越高；每张图像、每个类别最多保留置信度最高的100个预测参与评分。

## 本地格式校验

安装依赖：

```bash
pip install -r requirements_validator.txt
```

对正式A榜结果执行：

```bash
python tools/validate_submission.py \
  --submission result.json \
  --image_dir /实际解压路径/测试集A/testA/images
```

`example_result_rle.json`仅使用训练图像展示一个合法非空RLE实例，不是A榜正式提交文件，也不包含测试集答案。如需检查该示例，可把训练图像目录传给校验器并增加`--example`参数。

常见拒绝原因包括：JSON损坏或不是UTF-8、缺少或重复图像、加入额外字段、类别不是整数0/1、分数超出范围、RLE尺寸不符、`counts`无法解码或掩码为空。
