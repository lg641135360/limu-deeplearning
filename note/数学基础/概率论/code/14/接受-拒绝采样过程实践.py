import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm


# p(z):目标采样分布
def p(z):
    return (0.3 * np.exp(-(z - 0.3) ** 2) +
            0.7 * np.exp(-(z - 2.) ** 2 / 0.3)) / 1.2113


# q建议分布
q_norm_rv = norm(loc=1.4, scale=1.2)
M = 2.5

# 均匀分布：接受采样判定环节使用
uniform_rv = uniform(loc=0, scale=1)
# 采样
z_samples = []

for i in range(100000):
    # 采集
    # 从建议分布q中采样得到样本值z
    z = q_norm_rv.rvs(1)[0]
    # 从均匀分布中采样得到u
    u = uniform_rv.rvs(1)[0]
    # 接受采样的判定
    # 当u满足条件加入样本集
    if p(z) >= u * M * q_norm_rv.pdf(2):
        z_samples.append(z)
# 绘制
x = np.arange(-3., 5., 0.01)
plt.gca().axes.set_xlim(-3, 5)
# 真实的目标采样
plt.plot(x, p(x), lw=3, color='r')
# 频率直方图 看拟合度
plt.hist(z_samples, alpha=0.6, bins=150, density=True, edgecolor='k')
plt.grid(ls='--')
plt.show()
