import numpy as np

A = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.4, 0.4]])


def get_matrix_pow(n):
    ret = A
    for i in range(n):
        ret = np.dot(ret, A)
    print(ret)

get_matrix_pow(3)
get_matrix_pow(5)
get_matrix_pow(10)
get_matrix_pow(20)
get_matrix_pow(100)