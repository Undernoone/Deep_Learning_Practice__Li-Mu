import torch
from torch import nn
from d2l import torch as d2l

# Attention Mechanism
# Nadaraya-Watson kernel

# Generate data
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5) # ,_ 抛弃第二个返回值即索引

# Target function
def f(x):
    return 2 * torch.sin(x) + x ** 0.8

# 添加噪声
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))

# Test data
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
x_test_len = len(x_test)
print(x_test_len)

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth','Pred'],
             xlim=[0,5], ylim=[-1,5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)

# 平均预测结果
# maan计算平均值
y_hat = torch.repeat_interleave(y_train.mean(), x_test_len)
plot_kernel_reg(y_hat)
d2l.plt.show()

# 非参数注意力汇聚，沐神将Pooling不叫池化叫汇聚
X_test_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
attention_weights = nn.functional.softmax(-(X_test_repeat - x_train) ** 2 / 2, dim=1)
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
d2l.plt.show()

# 可视化注意力权重
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs', ylabel='Sorted test inputs')
d2l.plt.show()


weights = torch.ones((2,10)) * 0.1
values = torch.arange(20.0).reshape((2,10))
print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))

# 带参数的注意力汇聚
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,),requires_grad=True))

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape(-1,keys.shape[1])
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w)**2/2,dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1),values.unsqueeze(-1)).reshape(-1)
X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train,-1))
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape(n_train, -1)

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch',ylabel='loss',xlim=[1,5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train) / 2
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch+1}, loss {float(l.sum()):.6f}')
    animator.add(epoch+1, float(l.sum()))
d2l.plt.show()

keys = x_train.repeat((x_test_len, 1))
values = y_train.repeat((x_test_len, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
d2l.plt.show()
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs', ylabel='Sorted testing inputs')
d2l.plt.show()