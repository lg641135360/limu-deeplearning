from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法 -》 tensor数据类型
# 通过 transforms.toTensor看两个问题
# 1. transforms如何使用
# 2. 为何需要Tensor数据类型

img_path = "data/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
# print(img)

writer = SummaryWriter("log")

# 1. transforms如何使用
tensor_trans = transforms.ToTensor()  # 返回一个trans的对象
tensor_img = tensor_trans(img)  # command + p提示传入参数

print(tensor_img)

# 2. 为何需要Tensor数据类型
writer.add_image("Tensor_img",tensor_img)

writer.close()