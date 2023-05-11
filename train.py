import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import cv2
from PIL import Image


batch_size = 4
learning_rate = 1e-4
epoches = 100

num_type = {0:"blight",
            1:"common_rust",
            2:"gray_leaf_spot",
            3:"healthy"}

# 加载数据集
import torchvision.datasets as dsets
import torchvision.transforms as transforms

trainpath = './dataset/train/'
valpath = './dataset/val/'


traintransform = transforms.Compose([
    transforms.RandomRotation(20),  # optional
    transforms.ColorJitter(brightness=0.1),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

valtransform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
])

trainData = dsets.ImageFolder(trainpath, transform=traintransform)  # 读取训练集，标签就是train目录下的文件夹的名字，图像保存在格子标签下的文件夹里
valData = dsets.ImageFolder(valpath, transform=valtransform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=batch_size, shuffle=False)
val_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(valpath))])
train_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(trainpath))])

# 定义模型 以调用最简单的torchvision自带的resnet34为例

model = models.resnet34(weights=None) #weights表示是否加载
model.fc = torch.nn.Linear(512, 4) #将最后的fc层的输出改为标签数量（如3）,512取决于原始网络fc层的输入通道
model = model.cuda()  # 如果有GPU，而且确认使用则保留；如果没有GPU，请删除

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 定义优化器

from torch.autograd import Variable

def train(model, optimizer, criterion):
    model.train()
    total_loss = 0
    train_corrects = 0
    for i, (image, label) in enumerate(trainLoader):
        image = Variable(image.cuda())  # 同理
        label = Variable(label.cuda())  # 同理
        optimizer.zero_grad()

        target = model(image)
        loss = criterion(target, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        max_value, max_index = torch.max(target, 1)
        pred_label = max_index.cpu().numpy()
        true_label = label.cpu().numpy()
        train_corrects += np.sum(pred_label == true_label)

    return total_loss / float(len(trainLoader)), train_corrects / train_sum


def evaluate(model, criterion):
    model.eval()
    corrects = eval_loss = 0
    with torch.no_grad():
        for image, label in testLoader:
            image = Variable(image.cuda())  # 如果不使用GPU，删除.cuda()
            label = Variable(label.cuda())  # 同理

            pred = model(image)
            loss = criterion(pred, label)

            eval_loss += loss.item()

            max_value, max_index = torch.max(pred, 1)
            pred_label = max_index.cpu().numpy()
            true_label = label.cpu().numpy()
            corrects += np.sum(pred_label == true_label)

    return eval_loss / float(len(testLoader)), corrects, corrects / val_sum


def predict():
    best_model_path = './model/bestmodel.pth'
    model = torch.load(best_model_path)
    model = model.cuda()

    # model = models.resnet34()
    # model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage), strict=True) # 利用cpu进行测试

    model.eval()

    testLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=1, shuffle=False)

    for i, (image, label) in enumerate(testLoader):
        image = Variable(image.cuda())  # 如果不使用GPU，删除.cuda()
        # image = Variable(image)
        pred = model(image)
        max_value, max_index = torch.max(pred, 1)
        pred_label = max_index.cpu().numpy()
        print(max_value, pred_label)
        # todo 预测显示
        # text = num_type[pred_label[0]]
        # img = transforms.ToPILImage(image)
        # img.show()
        # img = image.cpu().numpy()  # FloatTensor转为ndarray
        # img_show = np.squeeze(img, 0)
        # img_show = np.transpose(img_show, (1, 2, 0))  # 把channel那一维放到最后
        # # 显示图片
        # plt.imshow(img_show)
        # plt.show()

        # img = cv2.putText(img, text, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100, 200, 200), 5)
        # cv2.imshow("output", img)
        # cv2.waitKey(0)
        """
        probs = F.softmax(pred, dim=1)
        # print("Sample probabilities: ", probs[:2].data.detach().cpu().numpy())
        a, b = np.unravel_index(probs[:2].data.detach().cpu().numpy().argmax(),
                                probs[:2].data.detach().cpu().numpy().shape)  # 索引最大值的位置   ###b就说预测的label
        print(testLoader.dataset.imgs[i][0])

        print('预测结果的概率:', round(probs[:2].data.detach().cpu().numpy()[0][b] * 100))
        print("label:  "+str(b))
        """


def main():
    train_loss = []
    valid_loss = []
    accuracy = []
    save_model_path = './model/'
    bestacc = 0

    for epoch in range(1, epoches + 1):
        epoch_start_time = time.time()
        loss, train_acc = train(model, optimizer, criterion)

        train_loss.append(loss)
        print('| start of epoch {:3d} | time: {:2.2f}s | train_loss {:5.6f}  | train_acc {}'.format(epoch, time.time() - epoch_start_time, loss, train_acc))

        loss, corrects, acc = evaluate(model, criterion)

        valid_loss.append(loss)
        accuracy.append(acc)

        if acc > bestacc:
            torch.save(model, save_model_path + 'bestmodel.pth')
            bestacc = acc

        print('| end of epoch {:3d} | time: {:2.2f}s | test_loss {:.6f} | accuracy {}'.format(epoch, time.time() - epoch_start_time, loss, acc))


    print("**********ending*********")
    # plt.plot(train_loss)
    # plt.plot(valid_loss)
    # plt.title('loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig("./loss.jpg")
    # # plt.show()
    # plt.cla()
    # plt.plot(accuracy)
    # plt.title('acc')
    # plt.ylabel('acc')
    # plt.xlabel('epoch')
    # plt.savefig("./acc.jpg")
    # plt.show()

if __name__ == '__main__':
    # main()
    predict()