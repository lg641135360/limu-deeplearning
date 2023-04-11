import scipy
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

s0 = 10.0
T = 1.0
n = 244 * T
mu = 0.15
sigma = 0.2
n_simulation = 100
dt = T / n

# 随机序列
random_series = np.zeros(int(n), dtype=float)
x = range(0, int(n))

# 针对每个样本
for i in range(n_simulation):
    random_series[0] = s0
    for j in range(1, int(n)):
        e = scipy.random.normal()
        random_series[j] = random_series[j - 1] + mu * random_series[j - 1] * dt + sigma * random_series[
            j - 1] * e * sigma * sqrt(dt)
    plt.plot(x, random_series)
plt.grid(ls='--')
plt.show()
