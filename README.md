# collaborative filtering

协同过滤推荐算法代码

## 应用对象

可用于用户-项目二部图中，向用户推荐未点击过的项目，并对结果进行评价（F1-score）。

## 使用方法

将训练集与测试集复制到data文件夹下，文件名为`train.txt`和`test.txt`，然后在目录下运行如下命令（注意是在命令行中，而不是在python代码中）：

```bash
python main.py --dataset data --type 1toA --sim c --nearest_user 10 --rec_item 100 --rec_result rec_result.txt
```

也可以采用如下方法【使用python代码运行】：

```python
import os

os.system("python main.py --dataset data --type 1toA --sim c --nearest_user 10 --rec_item 100 --rec_result rec_result.txt")
```

参数：

> --dataset=存放数据的文件夹，其中包含train.txt和test.txt。
>
> --type=数据的保存类型，分为：
>
> - 1to1:每行仅有两个元素，第一个为用户编号，第二个为项目编号；
> - 1to1_score:每行仅有三个元素，第一个为用户编号，第二个为项目编号，第三个为权重；
> - 1toA:每行包含多个元素，第一个为用户编号，其余为项目编号（默认）。
>
> （项目默认data文件夹下存有amazon-book数据，属于1to1的文件结构。）
>
> --sim=相似度计算方式，包括：c:余弦相似度（默认）；t:tanimoto相似度； l:对数似然比； m:曼哈顿距离
>
> --nearest_user=为项目评分时参考的相似用户的数量
>
> --rec_item=最终推荐的项目数量
>
> --rec_result=最终推荐的项目结果保存路径（文本文件）

