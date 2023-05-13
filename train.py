import torch
import numpy as np
import os
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

'''-------------------------预设参数-------------------------'''
batch_size = 4
learning_rate = 1e-4  # 学习率
epoches = 100  # 训练轮次
train_path = './dataset/train/'  # 训练集的路径
val_path = './dataset/val/'  # 验证集的路径
use_gpu = True  # 是否使用gpu

'''-------------------------数据加载-------------------------'''
# 设定训练数据集的预处理方式
train_transform = transforms.Compose([
    transforms.RandomRotation(20),  # 随机旋转
    transforms.ColorJitter(brightness=0.1),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
])

# 设定验证数据集的预处理方式
val_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # 将图片数据变为tensor格式
])

trainData = dsets.ImageFolder(train_path, transform=train_transform)  # 读取训练集，标签是train目录下的文件夹名称(0，1，2，3)
valData = dsets.ImageFolder(val_path, transform=val_transform)  # 读取验证集，标签是test目录下的文件夹名称(0，1，2，3)

# 按batch_size打包数据集
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=valData, batch_size=batch_size, shuffle=False)

# 记录训练集和验证集的总数，用于后面计算准确率
val_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(val_path))])
train_sum = sum([len(x) for _, _, x in os.walk(os.path.dirname(train_path))])

'''-------------------------模型定义-------------------------'''
model = models.resnet34(weights=None)  # 模型用resnet34
model.fc = torch.nn.Linear(512, 4)  # 将最后的fc层的输出改为标签数量（4）,512取决于原始网络fc层的输入通道
if use_gpu:
    model = model.cuda()  # 使用gpu进行加速

criterion = torch.nn.CrossEntropyLoss()  # 损失函数用交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器用Adam


def train(model, optimizer, criterion):
    """
    模型训练
    :param model: 模型
    :param optimizer: 优化器
    :param criterion:
    :return:
    """
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
            if use_gpu:
                image = Variable(image.cuda())
                label = Variable(label.cuda())
            else:
                image = Variable(image)
                label = Variable(label)

            pred = model(image)
            loss = criterion(pred, label)

            eval_loss += loss.item()

            max_value, max_index = torch.max(pred, 1)
            pred_label = max_index.cpu().numpy()
            true_label = label.cpu().numpy()
            corrects += np.sum(pred_label == true_label)

    return eval_loss / float(len(testLoader)), corrects, corrects / val_sum


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

        if epoch % 10 == 0:
            torch.save(model, save_model_path + 'model' + str(epoch) + '.pth')
        if acc > bestacc:
            torch.save(model, save_model_path + 'bestmodel.pth')
            bestacc = acc

        print('| end of epoch {:3d} | time: {:2.2f}s | test_loss {:.6f} | accuracy {}'.format(epoch, time.time() - epoch_start_time, loss, acc))


    print("**********ending*********")
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./loss.jpg")
    # plt.show()
    plt.cla()
    plt.plot(accuracy)
    plt.title('acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.savefig("./acc.jpg")
    plt.show()

if __name__ == '__main__':
    main()
