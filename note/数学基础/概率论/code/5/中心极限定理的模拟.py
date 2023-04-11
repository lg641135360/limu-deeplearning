import numpy as np
from scipy.stats import geom
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2)

# 生成几何分布
geom_rv = geom(p=0.3)
geom_rvs = geom_rv.rvs(size=1000000)
# 得到统计量，使用api
mean, var, skew, kurt = geom_rv.stats(moments='mvsk')
ax[0][0].hist(geom_rvs, bins=100, density=True)
ax[0][0].set_title('geom dist:p=0.3')
ax[0][0].grid(ls='--')

# 采样个数，三组实验的采样个数
n_array = [0, 2, 5, 50]

# 分别进行三组实验，1，2，3
for i in range(1, 4):
    Z_array = []  # 每次标准化的结果，放入最后结果
    n = n_array[i]  # 采样个数
    for j in range(100000):
        sample = np.random.choice(geom_rvs, n)  # 采样
        Z_array.append((sum(sample) - n * mean) / np.sqrt(n * var))
    ax[i // 2][i % 2].hist(Z_array, bins=100, density=True)
    ax[i // 2][i % 2].set_title('n={}'.format(n))
    ax[i // 2][i % 2].grid(ls='--')
plt.show()