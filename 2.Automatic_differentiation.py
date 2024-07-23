import torch
x = torch.arange(4.0)
print('x:',x) # 在外面计算y关于x的梯度之前，需要一个地方来存储梯度。

x.requires_grad_(True) # 等价于 x = torch.arange(4.0,requires_grad=True)
print('x.grad_1:',x.grad) # x.grad是存梯度的地方，默认为None，即还没有求导求出梯度出来

y = 2 * torch.dot(x,x) # 向量点积运算
print('y_2:',y) # grad_fn是隐式的构造了梯度函数，1+4+9=14
y.backward() # 反向传播后会有梯度计算出来
print('x.grad_2:',x.grad) # 访问导数，即访问梯度，y=2*x^2，y'=4x，所以x.grad=4x

x.grad.zero_() # 梯度清零
y = x.sum() # 这里的y是一个标量，sum函数是对张量所有的元素进行求和，即y=x1+x2+x3......，y'=1，所以x.grad自然是全1
print('y_3:',y)
y.backward()
print('x.grad_3:',x.grad)

y.backward()
x.grad.zero_()
y = x * x
print('y_4:',y) # 这里的y是一个向量，x^2，y'=2x，所以x.grad=2x
y.sum().backward()
print('x.grad_4:',x.grad)

x.grad.zero_()
y = x * x
print('y_5:',y)
u = y.detach() # y.detach把y当作一个常数，而不是关于x的一个函数
print('y.detach():',y.detach())
print('u:',u)
z = u * x
print('z:',z)
z.sum().backward()
print(x.grad == u) # 这里的x.grad是关于u的函数，所以x.grad(u) = u，即x.grad(u) = u(x) = u(2x) = 4x，所以x.grad(u) = 4x

def f(a):
    b = a * 2
    while b.norm() < 1000: # norm是L2范数
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(),requires_grad=True)
print('a:',a)
d = f(a)
d.backward()
print('a.grad:',a.grad)
print('d/a:',d/a)
print(a.grad == d/a) # d是a的线性函数，所以导数就是斜率d/a
