import torch
from torch.autograd import Variable

# part 1
#x = Variable(torch.ones(2, 2),
# requires_grad=True)

x = torch.ones(2, 2, requires_grad=True)

x.register_hook(lambda grad:grad*2)

y = x + 2
z = y * y * 3
# z = torch.sum(z)
# nn = torch.rand(2, 2)
nn = torch.ones(2, 2)
print(nn)


z.backward(gradient=nn, retain_graph=True)
torch.autograd.backward(z, grad_tensors=nn, retain_graph=True)

print(torch.autograd.grad(z, [x, y, z], grad_outputs=nn))

print(x.grad)
print(y.grad)
print(x.grad_fn)
print(y.grad_fn)
print(z.grad_fn)


# create a tensor and set requires_grad=True
x = torch.tensor([1.0, 2.0], requires_grad=True)
# Original tensor: tensor([1., 2.], requires_grad=True)
print("Original tensor:", x)

# tensor operation
y = x + 2  # add
z = y * y * 3  # multiplication
out = z.mean()  # mean

# compute gradients
out.backward()
print("Gradient of x with respect to the output:", x.grad)

'''
In some scenarios, such as when validating the model or calculating some intermediate results that do not require updating parameters, 
preventing gradient tracking can reduce memory consumption and improve efficiency. Using .detach() or torch.no_grad() are effective means of achieving this.
'''
# Using the .detach() method: Returns a new tensor with the same value as the original tensor, but does not track 
gradients.new_tensor = original_tensor.detach()

# Use torch.no_grad() context manager.
with torch.no_grad():
    # Operations performed in this area will not track gradients
    intermediate_result = some_operation(original_tensor)
