import random


# 抛掷一次硬币，出现正面向上的概率为p
def bernoulli_trial(p):  # u服从均匀分布，有p的概率返回1，1-p的概率返回0
    u = random.uniform(0, 1)
    if u <= p:
        return 1
    else:
        return 0


# 模拟抛掷硬币实验（伯努利实验）
def coin_experiments(n_arrary, p):
    y = 0  # 抛掷实验正面向上的次数
    n_max = max(n_arrary)  # [5,10,20,100,500,1000]获取的就是1000
    res = []
    for n in range(1, n_max + 1):
        y = y + bernoulli_trial(p)
        if n in n_arrary:  # 只需要添加[5,10,20,100,500,1000]的正面向上时的次数
            res.append((y, n))
    return res


print(coin_experiments([5, 10, 20, 100, 500, 1000], 0.62))
