import torch
from torch import nn
from d2l import torch as d2l

# 定义一个stride=1，padding=0的卷积层
def corr2d(X, K):
    """计算二维互相关运算。"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, X):
        return corr2d(X, self.weight) + self.bias

X = torch.ones((6, 8))
print(X)
X[:, 2:6] = 0
print(X)
# 简单应用一下，边缘检测（由0到1算-1边缘，由1到0算1边缘）
K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
X_t = X.t()
print(X)
print(Y) # 1代表1到0，-1代表0到1，所以输出的结果是边缘检测的结果
print(corr2d(X_t, K)) # X_t() 为X的转置，而K卷积核只能检测垂直边缘
# 学习由X生成Y的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1,2), bias=False) # 单个矩阵，输入通道为1，黑白图片通道为1，彩色图片通道为3。这里输入通道为1，输出通道为1.
X = X.reshape((1,1,6,8)) # 通道维：通道数，RGB图3通道，灰度图1通道，批量维就是样本维，就是样本数
Y = Y.reshape((1,1,6,7))
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad # 3e-2是学习率
    if(i+1) % 2 == 0:
        print(f'batch {i+1},loss {l.sum():.3f}')

# 所学的卷积核的权重张量
print(conv2d.weight.data.reshape((1,2)))