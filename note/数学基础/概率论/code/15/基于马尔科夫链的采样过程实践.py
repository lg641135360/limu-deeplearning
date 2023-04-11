import numpy as np
from scipy.stats import uniform
import random


# 当前状态、转移状态矩阵
# 返回转移后状态
def randomstate_gen(cur_stat, transfer_matrix):
    uniform_rvs = uniform().rvs(1)  # 均匀分布，生成一个随机数
    i = cur_stat - 1
    if uniform_rvs[0] <= transfer_matrix[i][0]:
        return 1
    elif uniform_rvs[0] <= transfer_matrix[i][0] + transfer_matrix[i][1]:
        return 2
    else:
        return 3


transfer_matrix = np.array([[0.7, 0.1, 0.2],
                            [0.3, 0.5, 0.2],
                            [0.1, 0.3, 0.6]], dtype='float32')
m = 10000
N = 100000
# 任取一个初始状态
cur_state = random.choice([1, 2, 3])
state_list = []

for i in range(m + N):
    state_list.append(cur_state)
    cur_state = randomstate_gen(cur_state, transfer_matrix)

# 切片，保留m之后的N的状态
state_list = state_list[m:]
print(state_list.count(1) / float(len(state_list)))
print(state_list.count(2) / float(len(state_list)))
print(state_list.count(3) / float(len(state_list)))
