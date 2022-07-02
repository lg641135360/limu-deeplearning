import torch

outputs = torch.tensor([[0.1, 0.2],
                        [0.5, 0.4]])

print(outputs.argmax(1))  # 返回较大的值的索引，行向量间比较(1)
preds = outputs.argmax(1)
targets = torch.tensor([0, 1])
print((preds == targets).sum().item())  # 计算预测正确的个数
