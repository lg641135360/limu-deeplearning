import numpy as np
from hmmlearn import hmm

# 隐含状态集Q
states = ['box1', 'box2', 'box3']
# 观测集合V
observations = ['black', 'white']
# 初始概率Pi
start_probability = np.array([0.3, 0.5, 0.2])
# 转移概率矩阵 A
transition_probability = np.array([
    [0.4, 0.4, 0.2],
    [0.3, 0.2, 0.5],
    [0.2, 0.6, 0.2]
])
# 观测概率矩阵B
emission_probability = np.array([
    [0.2, 0.8],
    [0.6, 0.4],
    [0.4, 0.6]
])

# 利用api建模离散观测状态初始化
model = hmm.MultinomialHMM(n_components=len(states))
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

observation_list = np.array([0, 1, 0])
# 打印概率的估计值
print(model.score(observation_list.reshape(-1, 1)))
