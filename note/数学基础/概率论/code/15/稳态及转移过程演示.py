import numpy as np
import matplotlib.pyplot as plt

transfer_matrix = np.array([[0.7, 0.1, 0.2],
                            [0.3, 0.5, 0.2],
                            [0.1, 0.3, 0.6]], dtype='float32')

# 三组不同实验
start_state_array = np.array([[0.50, 0.30, 0.20],
                              [0.13, 0.28, 0.59],
                              [0.10, 0.85, 0.05]], dtype='float32')
trans_step = 10

# 转移
# pi_t+1 = pi_t *p
for i in range(3):
    state_1_value = []
    state_2_value = []
    state_3_value = []
    # 转移10步
    for _ in range(trans_step):
        start_state_array[i] = np.dot(start_state_array[i], transfer_matrix)
        # 记录转移后的概率
        state_1_value.append(start_state_array[i][0])
        state_2_value.append(start_state_array[i][1])
        state_3_value.append(start_state_array[i][2])
    x = np.arange(trans_step)
    plt.plot(x, state_1_value, label='state_1')
    plt.plot(x, state_2_value, label='state_2')
    plt.plot(x, state_3_value, label='state_3')
    plt.legend()

    print(start_state_array[i])
plt.gca().axes.set_xticks(np.arange(0, trans_step))  # x刻度
plt.gca().axes.set_yticks(np.arange(0.2, 0.6, 0.05))
plt.show()
