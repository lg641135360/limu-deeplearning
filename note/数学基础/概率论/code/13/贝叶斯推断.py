import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn

seaborn.set()

# 参考结果
theta_real = 0.62

# 观测数据
# 抛掷次数
n_array = [5, 10, 20, 100, 500, 1000]
# 正面朝上的次数
y_array = [2, 4, 11, 60, 306, 614]

# beta分布参数列表
beta_params = [(0.25, 0.25), (1, 1), (10, 10)]
x = np.linspace(0, 1, 100)

# 绘制两行三列，共享xy轴
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

# 绘制
for i in range(2):
    for j in range(3):
        # 观测数据
        n = n_array[3 * i + j]
        y = y_array[3 * i + j]
        # 计算beta后验分布参数
        # 打包后(0.25,0.25,'b')...
        for (a_prior, b_prior), c in zip(beta_params, ('b', 'r', 'g')):
            a_post = a_prior + y
            b_post = b_prior + n - y
            # 绘制pdf曲线 后验，beta传参为x坐标，两个beta分布参数
            p_theta_given_y = beta.pdf(x, a_post, b_post)
            ax[i, j].plot(x, p_theta_given_y, c)
            ax[i, j].fill_between(x, 0, p_theta_given_y, color=c, alpha=0.25)
        # 画真实值对比
        # 画成一根竖线
        ax[i,j].axvline(theta_real,ymax=0.5,color='k')
        ax[i,j].set_xticks([0,0.2,0.4,0.6,0.8,1]) # 设置x刻度
        ax[i,j].set_title('n={},y={}'.format(n,y))
plt.show()