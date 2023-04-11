import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# p(z)：目标采样分布
def p(z):
    return (0.3 * np.exp(-(z - 0.3) ** 2) +
            0.7 * np.exp(-(z - 2.) ** 2 / 0.3)) / 1.2113


# 建议分布q
q_norm_rv = norm(loc=1.4, scale=1.2)
# M
M = 2.5
# 绘制p(z)和q(z)到一张图
z = np.arange(-4., 6., 0.01)
plt.plot(z, p(z), color='r', lw=3, alpha=0.6, label='p(z)')
plt.plot(z, M * q_norm_rv.pdf(z), color='b', lw=3, alpha=0.6, label='Mq(z)')
plt.legend()
plt.grid(ls='--')
plt.show()
