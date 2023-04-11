import numpy as np
import matplotlib.pyplot as plt

#
# mean = np.array([0, 0])  # 均值
# conv = np.array([[1, 0], [0, 1]])  # 协方差矩阵
#
# # 生成样本 多元正态分布的
# x, y = np.random.multivariate_normal(mean=mean, cov=conv, size=5000).T  # 转置是api需求
#
# # 绘图操作
# plt.figure(figsize=(6, 6))  # 设置图片大小
# plt.plot(x, y, 'ro', alpha=0.2)
# plt.gca().axes.set_xlim(-4, 4)  # 设置x轴上下限
# plt.gca().axes.set_ylim(-4, 4)  # 设置y轴上下限
# plt.grid(ls='--')
# plt.show()


# mean = np.array([0, 0])
# conv_1 = np.array([[1, 0], [0, 1]])
#
# conv_2 = np.array([[4, 0], [0, 0.25]])
#
# # 构造两个不同的正态分布
# # 标准正态
# x_1, y_1 = np.random.multivariate_normal(mean=mean, cov=conv_1, size=3000).T
# x_2, y_2 = np.random.multivariate_normal(mean=mean, cov=conv_2, size=3000).T
#
# # 绘图
# plt.figure(figsize=(6, 6))
# plt.plot(x_1, y_1, 'ro', alpha=0.05)
# plt.plot(x_2, y_2, 'bo', alpha=0.05)
#
# # 设置坐标轴大小
# plt.gca().axes.set_xlim(-6, 6)
# plt.gca().axes.set_ylim(-6, 6)
#
# # 设置网格线
# plt.grid(ls='--')
# plt.show()

# 使用子图方式进行对比
fig, ax = plt.subplots(2, 2)
mean = np.array([0, 0])

conv_1 = np.array([[1, 0], [0, 1]])
conv_2 = np.array([[1, 0.3], [0.3, 1]])  # 方差不变，只改变协方差
conv_3 = np.array([[1, 0.85], [0.85, 1]])
conv_4 = np.array([[1, -0.85], [-0.85, 1]])

# 获取每组正态分布的3000组样本
x_1, y_1 = np.random.multivariate_normal(mean=mean, cov=conv_1, size=3000).T
x_2, y_2 = np.random.multivariate_normal(mean=mean, cov=conv_2, size=3000).T
x_3, y_3 = np.random.multivariate_normal(mean=mean, cov=conv_3, size=3000).T
x_4, y_4 = np.random.multivariate_normal(mean=mean, cov=conv_4, size=3000).T

# 绘制
ax[0][0].plot(x_1, y_1, 'bo', alpha=0.05)
ax[0][1].plot(x_2, y_2, 'bo', alpha=0.05)
ax[1][0].plot(x_3, y_3, 'bo', alpha=0.05)
ax[1][1].plot(x_4, y_4, 'bo', alpha=0.05)

# 背景网格线
ax[0][0].grid(ls='--')
ax[0][1].grid(ls='--')
ax[1][0].grid(ls='--')
ax[1][1].grid(ls='--')

plt.show()
