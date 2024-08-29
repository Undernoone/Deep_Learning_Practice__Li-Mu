import torch
from torch import nn

def comp_conv2d(conv2d,X):
    X = X.reshape((1,1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(in_channels=1,out_channels=1, # 输入通道数，输出通道数
                   kernel_size=3,padding=1,stride=2) # 卷积核大小，填充，步长
X = torch.rand(size=(8,8))
print(comp_conv2d(conv2d,X).shape)

conv2d = nn.Conv2d(1,1,(5,3),padding=(2,1))
print(comp_conv2d(conv2d,X).shape)

conv2d = nn.Conv2d(1,1,3,2,1)
print(comp_conv2d(conv2d,X).shape) # (8+2-3)/2取floor为4

conv2d = nn.Conv2d(1,1,(3,5),(3,4),(0,1))
print(comp_conv2d(conv2d,X).shape)