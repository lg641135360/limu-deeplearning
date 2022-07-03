import torchvision

from PIL import ImageDraw

# 字符串前加r代表不要转义该字符串
coco_dataset = torchvision.datasets.CocoDetection(root=r"/Users/rikoo/Documents/github/limu-deeplearning/object_detect/imgs",
                                                  annFile="/Users/rikoo/Documents/github/limu-deeplearning/object_detect/ann/myCOCO_labels.json")
# print(coco_dataset[0]) # 这里debug发现其返回两个参数
image, info = coco_dataset[0]
# image.show()

# 将框画出来
# 获取图片的句柄
image_handler = ImageDraw.ImageDraw(image)

for annotation in info:
    x_min, y_min, width, height = annotation['bbox']
    # print(x_min)
    # 传入框的左上角、右下角坐标
    image_handler.rectangle(((x_min, y_min), (x_min + width, y_min + height)))

image.show()
