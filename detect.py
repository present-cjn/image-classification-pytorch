import cv2
import torch
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

'''-------------------------预设参数-------------------------'''
# 类型代号和名称对应的字典(用于在图像显示，可自行修改)
num_type = {0: "blight",
            1: "common_rust",
            2: "gray_leaf_spot",
            3: "healthy"}
use_gpu = True  # 是否使用gpu
val_path = './dataset/val/'  # 指定检测用的数据集路径

'''-------------------------数据加载-------------------------'''
val_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
])
# 读取测试集，标签是目录下的文件夹名称(0，1，2，3)
valData = dsets.ImageFolder(val_path, transform=val_transform)


# 显示结果图的函数
def display(image, pred_label):
    array1 = image.cpu().numpy()  # 将tensor数据转为numpy数据
    array1 = np.squeeze(array1)  # 压缩第一个维度，将图片从[1, 3, 224, 224]变为[3, 224, 224]
    array1 = array1 * 255 / array1.max()  # 图像归一化
    mat = np.uint8(array1)  # 数据类型转换，float32-->uint8
    mat = mat.transpose(1, 2, 0)  # 调整图像的维度值(224, 224，3)，用于显示
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)  # opencv默认显示BGR，因此需要将mat从RGB变为BGR
    # 将预测结果写到图片上
    text = num_type[pred_label[0]]
    mat = cv2.putText(mat, text, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200), 1)
    # 图像显示
    cv2.imshow("img", mat)
    cv2.waitKey()


def predict():
    # 指定模型的路径
    best_model_path = './model/bestmodel.pth'

    # 加载模型(gpu或cpu都可以)
    if use_gpu:
        model = torch.load(best_model_path)
        model = model.cuda()
    else:
        model = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.eval()

    # 加载测试数据
    testLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=1, shuffle=False)

    # 迭代计算每张图片的预测值
    for i, (image, label) in enumerate(testLoader):
        if use_gpu:
            image = Variable(image.cuda())
        else:
            image = Variable(image)
        pred = model(image)
        max_value, max_index = torch.max(pred, 1)
        pred_label = max_index.cpu().numpy()
        print(max_value, pred_label)
        print(image.size())
        # 显示预测结果图
        display(image, pred_label)


if __name__ == '__main__':
    predict()