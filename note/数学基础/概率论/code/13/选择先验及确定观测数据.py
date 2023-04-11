import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn

seaborn.set()

x = np.linspace(0, 1, 100)

# 绘制beta分布为(0.25,0.25)的分布图
plt.plot(x, beta(0.25, 0.25).pdf(x), color='b', label='$\\alpha=0.25,\\beta=0.25$')
# 曲线和横轴之间填充(填充纵轴方向) 填充x轴，起点0，终点pdf曲线
plt.fill_between(x, 0, beta(0.25, 0.25).pdf(x), color='b', alpha=0.25)

plt.plot(x, beta(1, 1).pdf(x), color='g', label='$\\alpha=1,\\beta=1$')
plt.fill_between(x, 0, beta(1, 1).pdf(x), color='g', alpha=0.25)

plt.plot(x, beta(10, 10).pdf(x), color='r', label='$\\alpha=10,\\beta=10$')
plt.fill_between(x, 0, beta(10, 10).pdf(x), color='r', alpha=0.25)

# 绘图设置
plt.gca().axes.set_ylim(0, 10)
plt.legend()  # 图例，图中的每个曲线的名字
plt.show()
