from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("log")
img_path = "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)  # 此时已经变成numpy类型
print(type(img_array))
print(img_array.shape)  # (H,W,C) 高度、宽度、通道

writer.add_image("test", img_array, 1, dataformats='HWC')  # 第二个参数传入图片数据（tensor or numpy（opencv read就是）

# y = x
for i in range(100):
    writer.add_scalar("y=x", i, i)  # 添加标量数据的标题
writer.close()

# tensorboard --logdir=log --port=6007  指定打开端口显示日志
