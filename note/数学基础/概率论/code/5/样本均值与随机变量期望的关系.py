import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt

n = 10
p = 0.4
sample_size = 15000
expected_value = n * p
N_sample = range(1, sample_size, 10)  # 得到

for k in range(3):
    binom_rv = binom(n=n, p=p)  # 生成二项分布随机变量
    # 随着样本数目的增大，计算样本的均值和分布的均值
    X = binom_rv.rvs(size=sample_size)  # 生成15000个样本的随机变量
    sample_average = [X[:i].mean() for i in N_sample]  # 对随机变量求均值
    plt.plot(N_sample, sample_average, label='average of sample {}'.format(k))

# 画出真实期望的值 图
plt.plot(N_sample, expected_value * np.ones_like(sample_average), ls='--',
         label='true expected value:n*p={}'.format(n * p), c='k')
plt.legend()
plt.grid(ls='--')
plt.show()