import torch

# 陷阱1 解决方案：引入模型定义的文件
from save_model import *

# 方式1 加载模型
import torchvision.models
from torch import nn

model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2 加载模型参数
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))

# model = torch.load("vgg16_method2.pth")
# print(vgg16)


# 陷阱1 缺少模型的定义
# class Test_nn(nn.Module):
#     def __init__(self):
#         super(Test_nn, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x


# test_nn = Test_nn() # 不需要这一步
model = torch.load("test_nn.pth")
print(model)
