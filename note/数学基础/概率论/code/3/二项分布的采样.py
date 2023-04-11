from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

# fig, ax = plt.subplots(3, 1)
# params = [(10, 0.25), (10, 0.5), (10, 0.8)]
# x = range(0, 11)
# for i in range(len(params)):
#     binom_rv = binom(n=params[i][0], p=params[i][1])  # 生成对应参数的二项分布随机变量
#     rvs = binom_rv.rvs(size=100000)  # 重复采样100000次 返回数组--实验结果100000次的采样结果
#     ax[i].hist(rvs, bins=11, density=True, alpha=0.6, edgecolor='k')  # 直方图，11个直方图的条形，是否进行归一化
#     ax[i].set_title('n={},p={}'.format(params[i][0], params[i][1]))
#     ax[i].set_xlim(0, 10)
#     ax[i].set_ylim(0, 0.4)
#     ax[i].set_xticks(x)
#     ax[i].grid(ls='--')
#     print('rvs{}:{}'.format(i, rvs))  # 打印结果
# plt.show()


binom_rv = binom(n=10, p=0.25)
mean, var, skew, kurt = binom_rv.stats(moments='mvsk')  # 使用api获取

# 使用采样数据的样本均值和方差
binom_rvs = binom_rv.rvs(size=100000)
E_sam = np.mean(binom_rvs)
S_sam = np.std(binom_rvs)
V_sam = S_sam * S_sam

print('mean={},var={}'.format(mean, var))  # api
print('E_sam={},V_sam={}'.format(E_sam, V_sam))  # 样本
print('E=np={},V=np(1-p)={}'.format(10 * 0.25, 10 * 0.25 * 0.75))  # 公式
