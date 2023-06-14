import torch
from torch.nn import functional as F

x = torch.rand(2, 3)  # Random tensor of size (2, 3)
print(x)
x_softmax = F.softmax(x, dim=-1)  # Apply softmax along the last dimension
print(x_softmax)
print(x_softmax.sum(dim=-1))  # Verify that the sum along the last dimension is 1

# same rank different shape
x = torch.rand(1, 3)
y = torch.rand(3, 1)
print (x)
print (y)
# what is rank? The rank of a tensor is the number of indices required to uniquely select each element of the tensor
print(x + y)

# different rank

y = torch.rand(3)

print(x + 1)


