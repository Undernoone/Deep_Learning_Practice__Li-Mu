import numpy as np
import pandas as pd
import torch
import numpy
import os

# x = torch.arange(12)
# print(x)
# print(x.shape)
# print(x.numel())
# x_reshape = x.reshape(3, 4)
# print(x_reshape)
# print(x_reshape.shape)
#
# all_zeros = torch.zeros((2,3,4)) # 二维三行四列的全零张量
# print(all_zeros)
# all_ones = torch.ones((2,3,4)) # 二维三行四列的全一张量
# print(all_ones)
# print('------------------------------------------------------------')
#
# a = torch.tensor([1,2,3],)
# b = torch.tensor([4,5,6])
# print(a+b)
# print(torch.exp(a))
# print('------------------------------------------------------------')
#
# c = torch.arange(12, dtype=torch.float32,).reshape(3,4)
# d = torch.arange(12, 24, dtype=torch.float32,).reshape(3,4)
# print(c)
# print(d)
# print(torch.cat([c,d],dim=0)) # dim的数值为维度，0表示按行拼接，1表示按列拼接
# print(torch.cat([c,d],dim=1)) # dim的数值为维度，0表示按行拼接，1表示按列拼接
# print('------------------------------------------------------------')
#
# demo = [[1,2,3],[4,5,6]]
# demo_numpy = numpy.array(demo)
# demo_tensor = torch.tensor(demo)
# print(demo_numpy)
# print(demo_tensor)
# print(type(demo_numpy))
# print(type(demo_tensor))
# print('------------------------------------------------------------')
#
# e = torch.tensor([3.5])
# print(e)
# print(e.item())
# print(type(e))
# print(type(e.item())) # 转换为python的float类型

# 创建一个名为“李沐深度学习data”的文件夹，exist_ok=True表示如果文件夹已存在不会报错
os.makedirs(os.path.join("..", "李沐深度学习data"), exist_ok=True) # 创建文件夹

# 定义csv文件的路径
data_file = os.path.join("..", "李沐深度学习data", "data.csv") # 文件路径

# 写入数据到csv文件中
with open(data_file, "w") as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('3,NA,178100\n')
    f.write('4,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)
print(outputs)
inputs['NumRooms'] = pd.to_numeric(inputs['NumRooms'], errors='coerce')
numeric_columns = inputs.select_dtypes(include=[np.number]).columns
inputs[numeric_columns] = inputs[numeric_columns].fillna(inputs[numeric_columns].mean())
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

inputs = inputs.apply(pd.to_numeric)
inputs_tensor = torch.tensor(inputs.values, dtype=torch.float32)
outputs_tensor = torch.tensor(outputs.values, dtype=torch.float32)
print(inputs_tensor)
print(outputs_tensor)