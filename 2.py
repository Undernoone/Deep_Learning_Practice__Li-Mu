import torch

x = torch.tensor([1, 2, 3])
print(x)

x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)