import numpy as np

mean = np.array([0, 0])
conv = np.array([[1, 0.85], [0.85, 1]])

x_1, y_1 = np.random.multivariate_normal(mean=mean, cov=conv, size=3000).T
x_2 = x_1 * 100
y_2 = y_1 * 100

print(np.corrcoef(np.vstack((x_1, y_1)))) # 计算相关系数，同样也需要堆叠向量
print(np.corrcoef(np.vstack((x_2, y_2))))
