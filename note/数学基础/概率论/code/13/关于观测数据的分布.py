from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
import seaborn

seaborn.set()

n = 10
p_params = [0.35, 0.5, 0.8]
x = np.linspace(0, n + 1)
f, ax = plt.subplots(3, 1)

for i in range(3):
    p = p_params[i]
    y = binom(n=n, p=p).pmf(x)
    ax[i].vlines(x, 0, y, colors='red', lw=10)  # 画垂线
    ax[i].set_ylim(0, 0.5)
    ax[i].plot(0, 0, label='n={}\n$\\theta$={}'.format(n, p), alpha=0.2)
    ax[i].legend()  # 显示
    ax[i].set_xticks(x)  # 横坐标刻度值
plt.show()
