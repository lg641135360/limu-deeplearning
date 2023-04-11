import pandas as pd
import random

sample_list = []
person_num = 100
round_num = 10000

for person in range(1, person_num + 1):
    money = 10
    for round in range(1, round_num + 1):
        res = random.randint(0, 1)  # 等概率生成0，1
        if res == 1:
            money = money + 1  # 赌赢了
        elif res == 0:
            money = money - 1  # 赌输了
        if money == 0:
            break
    sample_list.append([person, round, money])  # 赌局结果加入到结果集

# 构造成表格形式
sample_df = pd.DataFrame(sample_list, columns=['person', 'round', 'money'])
# 将person作为索引，并且不需要该列
sample_df.set_index('person', inplace=True)
# 打印结果分析
print('总轮数：{}，总人数：{}'.format(round_num, person_num))
# 总数-赌满人数
print('输完赌本提前出局的人数：{}'.format(person_num - len(sample_df[sample_df['round'] == round_num])))
# 赌满全场且盈利的人数
# 使用条件索引
print('赌满全场且盈利的人数：{}'.format(len(sample_df[sample_df['money'] > 10])))
# 赌满全场且亏本的人数
print('赌满全场且亏本的人数:{}'.format(len(sample_df[sample_df['money'] <= 10])))
