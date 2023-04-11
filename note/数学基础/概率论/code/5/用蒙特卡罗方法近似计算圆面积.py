import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import uniform

n = 100000
r = 1.0
o_x, o_y = (0., 0.)  # 圆心横纵坐标 使用元组进行赋值

# 生成100000均匀分布点的横坐标
uniform_x = uniform(o_x - r, 2 * r).rvs(n)  # 起点从圆心-r，跨度则是直径，生成100000个
uniform_y = uniform(o_y - r, 2 * r).rvs(n)

# 距离数组，均匀分布的点与圆心距离，小于半径在圆内
d_array = np.sqrt((uniform_x - o_x) ** 2 + (uniform_y - o_y) ** 2)
# 小于半径 赋1，不然赋为0，这里求和，使用np的方法，小于半径的就是1，加起来所有在园内的点个数
res = sum(np.where(d_array < r, 1, 0))
pi = (res / n) / (r ** 2) * (2 * r) ** 2

fig, ax = plt.subplots(1, 1)
ax.plot(uniform_x, uniform_y, 'ro', alpha=0.2, markersize=0.3)
plt.axis('equal')
circle = Circle(xy = (o_x,o_y),radius=r,alpha=0.5)
ax.add_patch(circle)

print('pi={}'.format(pi))
plt.grid(ls='--')
plt.show()
