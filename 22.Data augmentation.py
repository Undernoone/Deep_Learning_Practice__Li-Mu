import torch
import torchvision
from torch import nn
from d2l import torch as d2l


d2l.set_figsize()
img_path = 'D:\\Deeplearning_Li Mu_Date\\img\\cat1.jpg'
img = d2l.Image.open(img_path)
d2l.plt.imshow(img)
d2l.plt.show()

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5): # 传入aug图片增广方法
    Y = [aug(img) for _ in range(num_rows * num_cols)] # 用aug方法对图片作用八次
    d2l.show_images(Y, num_rows, num_cols, scale=scale) # 生成结果用num_cols行，num_cols列展示

apply(img, torchvision.transforms.RandomHorizontalFlip()) # 水平随机翻转
d2l.plt.show()