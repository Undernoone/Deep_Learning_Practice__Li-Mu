import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]): # Loop over rows
        for j in range(Y.shape[1]): # Loop over columns
            if mode == 'max':
                Y[i,j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i,j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
print(pool2d(X, (2,2))) # max
print(pool2d(X, (2,2), 'avg')) # avg

# Pytorch中的池化层
X = torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
print(X)
pool2d_torch = nn.MaxPool2d(3) # 深度学习框架中的步幅默认与池化窗口的大小相同，下一个窗口和前一个窗口没有重叠的
print(pool2d_torch(X))

# 填充和步幅可以手动设定
pool2d = nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))

# 设定一个任意大小的矩形池化窗口，并分别设定填充和步幅的高度和宽度
pool2d = nn.MaxPool2d((2,3),padding=(1,1),stride=(2,3))
print(pool2d(X))

# 池化层在每个通道上单独运算
X = torch.cat((X,X+1),1)
print(X.shape) # 合并起来，变成了1X2X4X4的矩阵
print(X)

pool2d = nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d(X))