import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# 类型代号和名称对应的字典(用于在图像显示，可自行修改)
num_type = {0: "blight",
            1: "common_rust",
            2: "gray_leaf_spot",
            3: "healthy"}

# 指定检测用的数据集路径
val_path = './dataset/val/'

val_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
])

valData = dsets.ImageFolder(val_path, transform=val_transform)  # 读取验证集，标签是test目录下的文件夹名称(0，1，2，3)


def predict(use_gpu: bool = True):
    # 指定模型的路径
    best_model_path = './model/bestmodel.pth'

    # 加载模型(gpu或cpu都可以)
    if use_gpu:
        model = torch.load(best_model_path)
        model = model.cuda()
    else:
        model = torch.load(best_model_path, map_location=torch.device('cpu'))

    model.eval()

    testLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=1, shuffle=False)

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

        # 显示结果
        array1 = image.cpu().numpy()  # 将tensor数据转为numpy数据
        array1 = np.squeeze(array1)
        maxValue = array1.max()
        array1 = array1 * 255 / maxValue  # normalize，
        mat = np.uint8(array1)  # float32-->uint8
        print('mat_shape:', mat.shape)  # mat_shape: (3, 982, 814)
        mat = mat.transpose(1, 2, 0)  # mat_shape: (982, 814，3)
        mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
        text = num_type[pred_label[0]]
        mat = cv2.putText(mat, text, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200), 1)
        cv2.imshow("img", mat)
        cv2.waitKey()

        """
        probs = F.softmax(pred, dim=1)
        # print("Sample probabilities: ", probs[:2].data.detach().cpu().numpy())
        a, b = np.unravel_index(probs[:2].data.detach().cpu().numpy().argmax(),
                                probs[:2].data.detach().cpu().numpy().shape)  # 索引最大值的位置   ###b就说预测的label
        print(testLoader.dataset.imgs[i][0])

        print('预测结果的概率:', round(probs[:2].data.detach().cpu().numpy()[0][b] * 100))
        print("label:  "+str(b))
        """


if __name__ == '__main__':
    predict()