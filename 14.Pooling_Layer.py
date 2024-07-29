import torch
from torch import nn
from d2l import torch as d2l

# 实现池化层的正向传播
def pool2d(X, pool_size, mode='max'):  # 拿到输入，池化窗口大小
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)) # 输入的高减去窗口的高，再加上1，这里没有padding
    for i in range(Y.shape[0]): # 行遍历
        for j in range(Y.shape[1]): # 列遍历
            if mode == 'max':
                Y[i,j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i,j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

# 验证二维最大池化层的输出
X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
print(pool2d(X, (2,2)))

# 验证平均池化层
print(pool2d(X, (2,2), 'avg'))

# 填充和步幅
X = torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
print(X)
pool2d = nn.MaxPool2d(3) # 深度学习框架中的步幅默认与池化窗口的大小相同，下一个窗口和前一个窗口没有重叠的
pool2d(X)

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