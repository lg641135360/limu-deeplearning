import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1 : 模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2：参数当作字典保存（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# 陷阱1
class Test_nn(nn.Module):
    def __init__(self):
        super(Test_nn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


test_nn = Test_nn()
torch.save(test_nn, "test_nn.pth")
