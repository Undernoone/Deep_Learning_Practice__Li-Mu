import torch
from torch import nn
from d2l import torch as d2l

# Traditional Transpose Convolution
def transpose_conv2d(X, K):
    # 获取卷积核的宽度和高度
    h, w = K.shape # 卷积核的宽、高
    # 创建一个新的张量Y，其尺寸为输入X的尺寸加上卷积核K的尺寸减去1。在常规卷积中，输出尺寸通常是输入尺寸减去卷积核尺寸加1
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1)) # 正常的卷积后尺寸为(X.shape[0] - h + 1, X.shape[1] - w + 1)
    # 遍历输入张量X的每一行
    for i in range(X.shape[0]):
        # 遍历输入张量X的每一列
        for j in range(X.shape[1]):
            # 对于输入X的每一个元素，我们将其与卷积核K进行元素级别的乘法，然后将结果加到输出张量Y的相应位置上
            Y[i:i + h, j:j + w] += X[i, j] * K # 按元素乘法，加回到自己矩阵
    # 返回转置卷积的结果
    return Y
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
Y = transpose_conv2d(X, K)
print(Y)

# Pytorch Transpose Convolution
X, K = X.reshape(1,1,3,3), K.reshape(1,1,2,2)
tconv = nn.ConvTranspose2d(1,1,kernel_size=2,bias=False) # 输入通道数为1，输出通道数为1
tconv.weight.data = K
print(tconv(X))

tconv_p1 = nn.ConvTranspose2d(1,1,kernel_size=2,padding=1,bias=False) # 输入通道数为1，输出通道数为1
tconv_p1.weight.data = K
print(tconv_p1(X))

tconv_s2 = nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,bias=False) # 输入通道数为1，输出通道数为1
tconv_s2.weight.data = K
print(tconv_s2(X))
# stride此处可以理解成在输入每个元素之间补（stride-1）个0，然后按照stride=1的正常卷积进行计算
