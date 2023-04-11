from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

norm_rv = norm(loc=0, scale=1)
x = np.linspace(-1, 1, 1000)

sample_n = 100
x_array = []

for i in range(1000000):
    norm_rvs = norm_rv.rvs(size=sample_n)
    x_bar = sum(norm_rvs) / float(sample_n)
    x_array.append(x_bar)

print(np.mean(x_array))  # 样本均值的期望
plt.hist(x_array, bins=100, density=True, alpha=0.3, edgecolor='k')
plt.axvline(0, ymax=0.8, color='r')
plt.gca().axes.set_xlim(-0.4, 0.4)
plt.grid(ls='--')
plt.show()
