from scipy.stats import expon
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
expon_rv = expon()  # 默认参数 入=1
# plt.plot(x, expon_rv_0.pdf(x), color='r', lw=3, alpha=0.6, label='lambda=1')
# expon_rv_1 = expon(scale=2)  # scale = 1/ 入
# plt.plot(x, expon_rv_1.pdf(x), color='b', lw=3, alpha=0.6, label='lambda=0.5')
# plt.legend(loc='best')
# plt.show()
expon_rvs = expon_rv.rvs(100000)
plt.plot(x,expon_rv.pdf(x),color='r',lw=3,alpha=0.6)
plt.hist(expon_rvs,density=True,alpha=0.6,bins=50,edgecolor='k')
plt.grid('--')
plt.show()