import torch
import torchvision.transforms
from PIL import Image
from torch import nn

image_path = "../images/dog.png"
image = Image.open(image_path)

print(image)

# png 4通道，转换RGB
image = image.convert('RGB')
print(image)

# 定义要进行的操作：
# 1. 变换大小
# 2. 变换格式到tensor
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)


# 导入模型网络结构
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


model = torch.load("../test_Classify_gpu_9.pth", map_location=torch.device('cpu')) # gpu上训练的模型使用在cpu上需要映射
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)  # 不更新梯度，节约性能

# output = model(image)   # 需要一个4维的数据，第一个表示批量个数
print(output)

print(output.argmax(1))