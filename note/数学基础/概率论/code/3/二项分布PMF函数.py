from scipy.stats import binom
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 1)
params = [(10, 0.25), (10, 0.5), (10, 0.8)]
x = range(0, 11)  # 左闭右开

for i in range(len(params)):
    binom_rv = binom(n=params[i][0], p=params[i][1])
    ax[i].set_title('n={},p={}'.format(params[i][0], params[i][1]))
    ax[i].plot(x, binom_rv.pmf(x), 'bo', ms=8)
    ax[i].vlines(x, 0, binom_rv.pmf(x), colors='b', lw=3)
    ax[i].set_xlim(0, 10)
    ax[i].set_ylim(0, 0.35)
    ax[i].set_xticks(x)  # 横坐标刻度
    ax[i].set_yticks([0, 0.1, 0.2, 0.3])  # 纵坐标刻度
    ax[i].grid(ls='--')

plt.show()
