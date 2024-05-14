import torch
import numpy as np

# torch.tensor() is a class constructor that creates a tensor from a list or a NumPy array.
# The tensor data type is inferred from the input data type.
list_data = [1, 2, 3, 4]
tensor_from_list = torch.tensor(list_data)
print(tensor_from_list) # tensor([1, 2, 3, 4])

# torch.tensor() can also create a tensor from a NumPy array.
np_array = np.array([1, 2, 3])
tensor_from_np_tensor = torch.tensor(np_array)
print(tensor_from_np_tensor) # tensor([1, 2, 3], dtype=torch.int32)

a = torch.Tensor([[1, 2],[3, 4]])
print(a)
print(a.type)
np.array([[1, 2],[3, 4]])

b = torch.Tensor(2, 2)
print(b)

d = torch.tensor(((1, 2), (3, 4)))
print(d.type())
print(d.type_as(a))

d = torch.empty(2,3)
print(d.type())
print(d.type_as(a))

d = torch.zeros(2,3)
print(d.type())
print(d.type_as(a))

d = torch.zeros_like(d)
print(d.type())
print(d.type_as(a))

d = torch.eye(2, 2)
print(d.type())
print(d.type_as(a))

d = torch.ones(2, 2)
print(d.type())
print(d.type_as(a))

d = torch.ones_like(d)
print(d.type())
print(d.type_as(a))

d = torch.rand(2, 3)
print(d.type())
print(d.type_as(a))

d = torch.arange(2, 10, 2)
print(d.type())
print(d.type_as(a))

d = torch.linspace(10, 2, 3)
print(d.type())
print(d.type_as(a))

dd = torch.normal(mean=0, std=1, size=(2, 3), out=b)
print(b)
print(dd)

d = torch.normal(mean=torch.rand(5), std=torch.rand(5))
print(d.type())
print(d.type_as(a))

d = torch.Tensor(2, 2).uniform_(-1, 1)
print(d.type())
print(d.type_as(a))

d = torch.randperm(10)
print(d.type())
print(d.type_as(a))

