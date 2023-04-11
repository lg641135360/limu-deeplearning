from scipy.special import comb
import math


# C_n^head p_head
def get_possibility(n, head, p_head):
    return comb(n, head) * math.pow(p_head, head) * math.pow((1 - p_head), (n - head))


print(get_possibility(20, 13, 2 / 5))
print(get_possibility(20, 13, 1 / 2))
print(get_possibility(20, 13, 3 / 5))
