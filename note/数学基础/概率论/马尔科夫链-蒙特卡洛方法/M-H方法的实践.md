### M-H采样方法总结

* ![image-20230410162808007](M-H%E6%96%B9%E6%B3%95%E7%9A%84%E5%AE%9E%E8%B7%B5.assets/image-20230410162808007-1115288.png)
* ![image-20230410162817338](M-H%E6%96%B9%E6%B3%95%E7%9A%84%E5%AE%9E%E8%B7%B5.assets/image-20230410162817338.png)
* ![image-20230410162835901](M-H%E6%96%B9%E6%B3%95%E7%9A%84%E5%AE%9E%E8%B7%B5.assets/image-20230410162835901.png)

* 重复上述第二步～第三步采样过程m+N次，结束后，保留后N 次采样结果即可作为目标分布的近似采样

### M-H采样方法代码实践

![image-20230410162937326](M-H%E6%96%B9%E6%B3%95%E7%9A%84%E5%AE%9E%E8%B7%B5.assets/image-20230410162937326.png)

```python
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


# 采样分布pi
def pi(x):
    return (0.3 * np.exp(-(x - 0.3) ** 2) +
            0.7 * np.exp(-(x - 2.) ** 2 / 0.3)) / 1.2113


m = 10000  # 燃烧期
N = 100000  # 实际保留的有效样本数
sample = [0 for i in range(m + N)]  # 采样数组

sample[0] = 2  # 随机指定一个采样的起始点
for t in range(1, m + N):
    x = sample[t - 1]  # 获取当前已经得到的样本点x
    x_star = norm.rvs(loc=x, scale=1, size=1)[0]  # 生成下一个时刻采样点
    alpha = min(1, (pi(x_star) / pi(x)))  # 接受概率
    u = random.uniform(0, 1)
    if u < alpha:
        sample[t] = x_star
    else:
        sample[t] = x  # 原路折返
x = np.arange(-2, 4, 0.01)
plt.plot(x, pi(x), color='r', lw=3)  # 实际的目标分布PDF
plt.hist(sample[m:], bins=100, density=True, edgecolor='k') # 实际分布的近似采样
plt.grid(ls='--')
plt.show()
```

![image-20230410163738619](M-H%E6%96%B9%E6%B3%95%E7%9A%84%E5%AE%9E%E8%B7%B5.assets/image-20230410163738619.png)