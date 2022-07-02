# 搭建神经网络
import torch
from torch import nn


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


if __name__ == '__main__':
    test_Classify = Test_Classify()
    input = torch.ones((64, 3, 32, 32))
    output = test_Classify(input)
    print(output.shape)  # 返回64个数据，然后每个数据都有十个数字，每个数字代表属于每个类的概率
