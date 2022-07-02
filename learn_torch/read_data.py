import os.path

from torch.utils.data import Dataset
# import cv2
from PIL import Image  # 读取图片


# Dataset的职责
# 提供一种方式去获取数据以及label
# 1. 如何获取每个数据
# 2. 告诉总共有多少数据

# Dataloader的职责
# 为后面网络提供不同数据形式

class MyData(Dataset):  # 继承Dataset类
    # 初始化类，构造函数，创建类内变量并进行初始化
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 类中定义一个全局变量，可以传递给别的方法使用；这里相当于初始化成员变量
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)  # 连接图片路径字符串：图片路径+标签路径
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
