import torch
x = torch.arange(4.0)
print('x:',x)

x.requires_grad_(True) # Equivalent to x = torch.arange(4.0,requires_grad=True)
print('x.grad_1:',x.grad) # x.grad是存梯度的地方，默认为None

y = 2 * torch.dot(x,x) # y = 2（x^2） = 2(0+1+4+9) = 28
print('y_2:',y)
y.backward() # After this line, y.grad = 2x, where x is the input tensor
print('x.grad_2:',x.grad) # Accessing the derivative, accessing the gradient， y=2*（x^2），y'=4x，so,x.grad=4x

x.grad.zero_() # dereference the gradient
y = x.sum()
print('y_3:',y) # 这里的y是一个标量，0+1+2+3=6
y.backward()
print('x.grad_3:',x.grad) # Accessing the derivative, accessing the gradient， y=x1+x2+x3......，y'=1，so，x.grad=1

y.backward()
x.grad.zero_()
y = x * x
print('y_4:',y) # 这里的y是一个向量，y=x^2，y'=2x，所以x.grad=2x
y.sum().backward()
print('x.grad_4:',x.grad) # y=x^2，y'=2x，so，x.grad=2x

x.grad.zero_()
y = x * x
print('y_5:',y)
u = y.detach() # y.detach：将y当作一个常数，而不是关于x的一个函数
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
