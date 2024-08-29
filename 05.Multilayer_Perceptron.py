import torch
from torch import nn
from d2l import torch as d2l

# Traditional Multi-Layer Perceptron
# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#
# num_inputs, num_outputs, num_hiddens = 784, 10, 256
#
# W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
#
# params = [W1, b1, W2, b2]
#
# def relu(X):
#     a = torch.zeros_like(X)
#     return torch.max(X, a)
#
# def net(X):
#     X = X.reshape((-1, num_inputs))
#     H = relu(X @ W1 + b1)
#     return H @ W2 + b2
#
# loss = nn.CrossEntropyLoss()
#
# num_epochs, lr = 10, 0.1
# updater = torch.optim.SGD(params, lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
# d2l.plt.show()

# Pytorch Multi-Layer Perceptron
'''
nn.Sequential是PyTorch 中的一个容器，用于按顺序包装多个模块。数据会按照定义的顺序依次通过这些模块。
nn.Flatten() 是一个层，用于将输入展平为一维张量。
假设输入是一个形状为 (batch_size, C, H, W) 的图像，nn.Flatten() 会将每个图像展平为形状为 (batch_size, C * H * W) 的张量。
在这个例子中，假设输入是 28x28 的图像（即 784 个像素），展平后每个图像变成 784 个元素的一维张量。
nn.Linear(784, 256) 是一个全连接层（线性层），它将输入的 784 个元素映射到 256 个元素。
nn.ReLU() 是激活函数，它将所有负值变成 0。
nn.Linear(256, 10) 是另一个全连接层，它将输入的 256 个元素映射到 10 个元素。这个层用于最终的分类输出，假设有 10 个类别。
'''
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights) # net.apply 是 PyTorch 中的一个方法，它会遍历网络中的每一个层，并对每个层调用传入的函数。

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()