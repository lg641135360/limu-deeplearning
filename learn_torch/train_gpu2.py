import torch
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

# 引入model
# from model import *

# 记录时间
import time
 
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

# 定义训练设备
device = torch.device("cuda:1")

train_data = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 打印长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 占位符
print("训练数据集的长度为:{}", format(train_data_size))
print("训练测试集的长度为:{}", format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
class Test_Classify(nn.Module):
    def __init__(self):
        super(Test_Classify, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


test_Classify = Test_Classify()
# 网络模型转移到CUDA上
test_Classify.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 损失函数转移到CUDA上
loss_fn = loss_fn.to(device)

# 优化器 随机梯度下降
# learning_rate = 0.01
learning_rate = 1e-2  # 科学计数法
optimizer = torch.optim.SGD(test_Classify.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("log_train")
start_time = time.time()  

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))

    # 训练步骤开始
    test_Classify.train()  # 不太重要，不一定要调用，有的层需要，dropout之类的特殊层
    for data in train_dataloader:
        imgs, targets = data
        # 将数据转移到cuda
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = test_Classify(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # .item()将tensor数据类型转换成真实数字
        if total_train_step % 100 == 0:  # 遇100打印，避免无用信息
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数:{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    # 每训练一轮后再去评估其指标
    # 这里就不改变梯度，只用于测试，保证不进行调优
    test_Classify.eval()  # 针对特殊层的进行一些设置
    total_test_loss = 0
    total_accuracy = 0  # 比较预测正确个数
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = test_Classify(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy /
                      test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每一轮训练出的模型
    torch.save(test_Classify, "test_Classify_gpu_{}.pth".format(i))
    # 官方推荐保存方式
    # torch.save(test_Classify.state_dict(), "test_Classify_{}.pth".format(i))
    print("模型已保存")

writer.close()
