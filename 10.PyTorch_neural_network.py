from torch import nn
import torch
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.randn(2, 20)
print(net(X))
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

net = MLP(20, 256, 10)
print(net(X))

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))
#
# class FixedHiddenMLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.rand_weight = torch.rand((20,20),requires_grad=False)
#         self.linear = nn.Linear(20,20)
#
#     def forward(self, X):
#         X = self.linear(X)
#         X = F.relu(torch.mm(X, self.rand_weight + 1))
#         X = self.linear(X)
#         while X.abs().sum() > 1:
#             X /= 2
#         return X.sum()
#
# net = FixedHiddenMLP()
# X = torch.rand(2,20)
# net(X)







