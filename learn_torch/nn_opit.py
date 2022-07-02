import torch.optim
import torchvision
from torch import nn

from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Test_nn(nn.Module):
    def __init__(self):
        super(Test_nn, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 维持尺寸不变，padding应为2（kernel的一半）
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
test_nn = Test_nn()
optim = torch.optim.SGD(test_nn.parameters(), lr=0.01)  # 一开始较大的lr，后面调小学习率

# 训练所有数据20轮
for epoch in range(20):
    running_loss = 0.0
    # 对所有数据进行了一轮的学习
    for data in dataloader:
        imgs, targets = data
        outputs = test_nn(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()  # 重置优化器
        result_loss.backward()  # 不使用反向传播这个则不会给每个参数生成grad梯度
        optim.step()  # 根据反向传播得到的每个参数的梯度进行优化（调整
        running_loss = running_loss + result_loss
    print(running_loss)
