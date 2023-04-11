import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

norm_rvs = norm(loc=0, scale=20).rvs(size=1000000)
plt.hist(norm_rvs, density=True, alpha=0.3, color='b', bins=100, label='origin')

mean_array = []
for i in range(10000):
    sample = np.random.choice(norm_rvs, size=5, replace=False)  # 每次抽取5个不同的样本
    mean_array.append(np.mean(sample))
plt.hist(mean_array, density=True, alpha=0.3, color='r', bins=100, label='sample size=5')

mean_array = []
for i in range(10000):
    sample = np.random.choice(norm_rvs, size=50, replace=False)  # 每次抽取5个不同的样本
    mean_array.append(np.mean(sample))
plt.hist(mean_array, density=True, alpha=0.3, color='r', bins=100, label='sample size=50')

plt.gca().axes.set_xlim(-60, 60)
plt.legend(loc='best')
plt.grid(ls='--')
plt.show()
