# image-classification-pytorch

## 准备
安装环境

```shell
pip install -r requirements.txt
```

## 训练
训练代码为train.py
### 开始训练
根据需求修改代码中的“预设参数”部分(如下)
```python
batch_size = 4
learning_rate = 1e-4  # 学习率
epoches = 100  # 训练轮次
train_path = './dataset/train/'  # 训练集的路径
val_path = './dataset/val/'  # 验证集的路径
use_gpu = True  # 是否使用gpu
```

运行train.py开始训练

### 模型
训练的模型会保存在model文件夹下，目前的设置是每10个epoches会自动存一次，除此之外还会存一个表现最好的模型bestmodel.pth

### 训练数据
训练结束后会输出准确率acc和损失loss，并以图片形式存在当前目录下。

## 检测
检测代码为detect.py
根据需求修改代码中的“预设参数”部分(如下)
```python
# 类型代号和名称对应的字典(用于在图像显示，可自行修改)
num_type = {0: "blight",
            1: "common_rust",
            2: "gray_leaf_spot",
            3: "healthy"}
use_gpu = True  # 是否使用gpu
val_path = './dataset/val/'  # 指定检测用的数据集路径
```
运行detect.py可以查看检测结果
