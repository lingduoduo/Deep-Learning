import torch

dev = torch.device("cpu")
# dev = torch.device("cuda")
a = torch.tensor([2, 2],
                 dtype=torch.float32,
                 device=dev)
print(a)

i = torch.tensor([[0, 1, 2], [0, 1, 2]])
v = torch.tensor([1, 2, 3])
a = torch.sparse_coo_tensor(i, v, (4, 4),
                            dtype=torch.float32,
                            device=dev).to_dense()
print(a)


tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# index single element
element = tensor[0, 0]  # 1

# slice a row
slice_1 = tensor[0, :]  # [1, 2, 3]
slice_2 = tensor[:, 1]  # [2, 5]

# change element
tensor[0, 0] = 7
print(tensor)  # [[7, 2, 3], [4, 5, 6]]

# shape attribute
shape = tensor.shape  # torch.Size([2, 3])
print(shape)

# unsqueeze(dim) increase the dimension of the tensor by one
expanded_tensor = tensor.unsqueeze(0)  #  new shape = [1, 2, 3]
print(expanded_tensor)

# reshape(shape) reshpe the tensor to the given shape
reshaped_tensor = tensor.reshape(6)  # new shape = [6]
print(reshaped_tensor)
