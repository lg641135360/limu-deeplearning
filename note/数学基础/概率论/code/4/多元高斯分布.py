import numpy as np
import matplotlib.pyplot as plt

mean_1 = np.array([0, 0])
conv_1 = np.array([[1, 0], [0, 1]])

mean_2 = np.array([0, -7])
conv_2 = np.array([[4, 0], [0, 0.25]])

mean_3 = np.array([4, 4])
conv_3 = np.array([[4, -3], [-3, 0.25]])

x_1, y_1 = np.random.multivariate_normal(mean=mean_1, cov=conv_1, size=2000).T
x_2, y_2 = np.random.multivariate_normal(mean=mean_2, cov=conv_2, size=2000).T
x_3, y_3 = np.random.multivariate_normal(mean=mean_3, cov=conv_3, size=2000).T

plt.plot(x_1, y_1, 'ro', alpha=0.05)
plt.plot(x_2, y_2, 'bo', alpha=0.05)
plt.plot(x_3, y_3, 'go', alpha=0.05)

plt.gca().axes.set_xlim(-10, 10)
plt.gca().axes.set_ylim(-10, 10)
plt.grid(ls='--')
plt.show()
