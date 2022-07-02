import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

# 利用现有网络，改变其结构，当作前置结构，迁移学习
# 主干网络

# 加入一个线性层，让其做一个10分类（而不是原来的1000分类）
# vgg16_true.add_module("add_linear", nn.Linear(1000, 10))
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
