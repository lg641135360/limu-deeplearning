import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


# 采样分布pi
def pi(x):
    return (0.3 * np.exp(-(x - 0.3) ** 2) +
            0.7 * np.exp(-(x - 2.) ** 2 / 0.3)) / 1.2113


m = 10000  # 燃烧期
N = 100000  # 实际保留的有效样本数
sample = [0 for i in range(m + N)]  # 采样数组

sample[0] = 2  # 随机指定一个采样的起始点
for t in range(1, m + N):
    x = sample[t - 1]  # 获取当前已经得到的样本点x
    x_star = norm.rvs(loc=x, scale=1, size=1)[0]  # 生成下一个时刻采样点
    alpha = min(1, (pi(x_star) / pi(x)))  # 接受概率
    u = random.uniform(0, 1)
    if u < alpha:
        sample[t] = x_star
    else:
        sample[t] = x  # 原路折返
x = np.arange(-2, 4, 0.01)
plt.plot(x, pi(x), color='r', lw=3)  # 实际的目标分布PDF
plt.hist(sample[m:], bins=100, density=True, edgecolor='k') # 实际分布的近似采样
plt.grid(ls='--')
plt.show()