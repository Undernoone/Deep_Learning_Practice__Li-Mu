import random
import torch
import numpy as np
import torch.utils.data as data
from d2l import torch as d2l
from torch import nn

# '''
# 传统的线性回归算法
# idea:首先，我们设定了真实的权重 w 和偏置 b，并使用它们生成了 1000 个数据点。
# 接着，我们假设不知道这些真实值，通过训练模型来反推最接近真实值的权重和偏置。
# '''
# def date_generator(w,b,num_exaples):
#     """ 生成 y = Xw + b + 噪声 """
#     X = torch.normal(0,1,(num_exaples,len(w))) # 正态分布
#     print('X:',X)
#     y = torch.matmul(X,w) + b # 矩阵乘法
#     """
#      X 是一个形状为 (num_examples, len(w)) 的矩阵,此处len (w)=2
#      w 是一个形状为 (len(w), 1) 的列向量
#      阵乘法后得到形状为 (num_examples, 1) 的结果
#     """
#     y += torch.normal(0,0.01,y.shape) # 模拟噪声
#     return X, y.reshape((-1,1))
#
# true_w = torch.tensor([2,-3.4])
# true_b = 4.2
# features, labels = date_generator(true_w, true_b, 1000)
# """
#  使用真实的权重 true_w 和偏置 true_b 生成 1000 个样本的数据
#  synthetic_data 函数返回特征矩阵 features 和标签向量 labels
#  features 的形状是 (1000, len(true_w))，其中每一行是一个样本的特征
#  labels 的形状是 (1000, 1)，每一行是对应样本的标签
# """
# print('features:',features[0],'\nlabel:',labels[0])
#
# d2l.set_figsize()
# d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
# d2l.plt.show()
#
# def data_iter(batch_size,features,labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)
#     for i in range(0, num_examples, batch_size):
#         batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)]) # 当i+batch_size超出时，取num_examples
#         yield features[batch_indices], labels[batch_indices] # 获得随即顺序的特征，及对应的标签
#
# batch_size = 10
#
# for X,y in data_iter(batch_size, features, labels):
#     print(X, '\n', y) # 取一个批次后，就break跳出了
#     break
#
# # 定义初始化模型参数
# train_w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
# train_b = torch.zeros(1, requires_grad=True)
#
# # 定义模型
# def linreg(X,w,b):
#     return torch.matmul(X,w)+b
#
# def squared_loss(y_hat,y):
#     return (y_hat - y.reshape(y_hat.shape))**2/2 # 将y统一成与y_hat一样同尺寸
#
# def sgd(params,lr,batch_size):
#     with torch.no_grad(): # 不要产生梯度计算，减少内存消耗
#         for param in params:
#             param -= lr * param.grad / batch_size # 每个参数进行更新，损失函数没有求均值，所以这里除以 batch_size 求了均值。由于乘法的线性关系，这里除以放在loss的除以是等价的。
#             param.grad.zero_()
#
# lr = 0.03
# num_epochs = 3
# net = linreg # 这里用线性模型，这样写是很方便net赋予其他模型，只需要改一处，不需要下面所有网络模型名称都改
# loss = squared_loss
#
# for epoch in range(num_epochs):
#     for X,y in data_iter(batch_size,features,labels):
#         l = loss(net(X, train_w, train_b), y) # x和y的小批量损失
#         # 因为l是形状是(batch_size,1)，而不是一个标量。l中所有元素被加到一起
#         # 并以此计算关于[w,b]的梯度
#         l.sum().backward()
#         sgd([train_w, train_b], lr, batch_size) # 使用参数的梯度更新参数
#     with torch.no_grad():
#         train_l = loss(net(features, train_w, train_b), labels)
#         print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')
#
#     # 比较真实参数和通过训练学到的参数来评估训练的成功程度
#
# print(train_w)
# print(train_b)
# print(f'w的估计误差：{true_w - train_w.reshape(true_w.shape)}')
# print(f'b的估计误差：{true_b - train_b}')

# 使用PyTorch实现线性回归
def data_generator_pytorch(w,b,num_exaples):
    X = torch.normal(0,1,(num_exaples,len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = data_generator_pytorch(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

net = nn.Sequential(nn.Linear(2,1))

net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 5
for epoch in range(num_epochs):
    for X, y in load_array((features, labels), batch_size=10):
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch+1}, loss {float(l.mean()):f}')

