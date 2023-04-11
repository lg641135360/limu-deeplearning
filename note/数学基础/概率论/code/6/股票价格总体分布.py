import scipy
import matplotlib.pyplot as plt
from math import sqrt

s0 = 10.0
T = 1.0
n = 244 * T
mu = 0.15
sigma = 0.2

# 样本数
n_simulation = 10000
dt = T / n
s_array = []

# 模拟
for i in range(n_simulation):
    s = s0
    for j in range(int(n)):
        e = scipy.random.normal() # 核心 获取随机变量
        s = s + mu * s * dt + sigma * s * e * sqrt(dt)
    s_array.append(s)

# 频率直方图看分布
plt.hist(s_array, alpha=0.6, bins=30, density=True, edgecolor='k')
plt.grid(ls='--')
plt.show()
