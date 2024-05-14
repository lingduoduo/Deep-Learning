import torch

a = torch.rand(2, 2)
b = torch.rand(1, 2)
# a, 2*1
# b, 1*2
# c, 2*2
# 2*4*2*3
c = a + b
print(a)
print(b)
print(c)
print(c.shape)

# tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
# broadcasted_addition = [[2, 3, 4], [5, 6, 7]]
broadcasted_addition = tensor + torch.tensor([1, 1, 1])

