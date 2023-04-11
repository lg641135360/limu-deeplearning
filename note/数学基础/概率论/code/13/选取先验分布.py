import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn  # 绘图风格更好看

seaborn.set()

params = [0.25, 1, 10]
x = np.linspace(0, 1, 100)
f, ax = plt.subplots(3, 3, sharex=True, sharey=True)

for i in range(3):
    for j in range(3):
        a = params[i]
        b = params[j]
        y = beta(a, b).pdf(x)
        ax[i][j].plot(x, y, color='r')
        ax[i][j].set_title('$\\alpha$={},$\\beta={}$'.format(a, b))
        ax[i][j].set_ylim(0, 10)

ax[0][0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax[0][0].set_yticks([0, 2.5, 5, 7.5, 10])
plt.show()
