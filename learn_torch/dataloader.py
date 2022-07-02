import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

# batch_size 4 代表每次拿出4个数据然后打包成一个test_loader条目(turple)
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片以及 target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")

for epoch in range(2):  # 读取两次
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("test_data", imgs, step)
        step = step + 1

writer.close()
