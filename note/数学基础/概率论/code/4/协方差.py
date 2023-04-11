import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)
mean = np.array([0, 0])
conv = np.array([[1, 0.85], [0.85, 1]])

x_1, y_1 = np.random.multivariate_normal(mean=mean, cov=conv, size=3000).T
x_2 = x_1 * 100
y_2 = y_1 * 100

ax[0].plot(x_1, y_1, 'bo', alpha=0.05)
ax[1].plot(x_2, y_2, 'bo', alpha=0.05)

print(np.cov(np.vstack((x_1, y_1))))  # 使用np提供的计算conv的方法，传入元祖并且使用vstack
print(np.cov(np.vstack((x_2, y_2))))

ax[0].grid(ls='--')
ax[1].grid(ls='--')
plt.show()
