import numpy as np
import pandas as pd
import torch
import numpy
import os

x = torch.arange(12)
print('x:',x)
print('x.shape:',x.shape)
print(x.numel()) # returns the number of elements in the tensor(int)
x_reshape = x.reshape(3, 4)
print(x_reshape)
print(x_reshape.shape)
print(x.numel()) # returns the number of elements in the tensor(int)
print('------------------------------------------------------------')

all_zeros = torch.zeros((2,3,4)) # 2 dimension 3 row 4 col tensor with all zeros
print(all_zeros)
all_ones = torch.ones((2,3,4)) # 2 dimension 3 row 4 col tensor with all ones
print(all_ones)
print('------------------------------------------------------------')

a = torch.tensor([1,2,3],)
b = torch.tensor([4,5,6])
print(a+b) # [5 7 9]
print(torch.exp(a)) # [ 2.7182817  7.3890562  20.08553692]
print('------------------------------------------------------------')

c = torch.arange(12, dtype=torch.float32,).reshape(3,4)
d = torch.arange(12, 24, dtype=torch.float32,).reshape(3,4) # 左闭右开
print(c)
print(d)
print(torch.cat([c,d],dim=0)) # 维度0表示按行拼接，1表示按列拼接
print(torch.cat([c,d],dim=1)) #
print('------------------------------------------------------------')

demo = [[1,2,3],[4,5,6]] # List of lists
demo_numpy = numpy.array(demo)  # Convert list of lists to numpy array
demo_tensor = torch.tensor(demo) # Convert list of lists to tensor
print(demo_numpy)
print(demo_tensor)
print(type(demo))
print(type(demo_numpy))
print(type(demo_tensor))
print('------------------------------------------------------------')

e = torch.tensor([3])
print(e)
print(e.item()) # return the value of tensor as a Python scalar
print(type(e))
print(type(e.item())) # return the type of the scalar value
print('------------------------------------------------------------')

# 数据预处理
os.makedirs(os.path.join("..", "Deeplearning_Li Mu_Date"), exist_ok=True) # 创建文件夹
data_file = os.path.join("..", "Deeplearning_Li Mu_Date", "data.csv") # 文件路径

with open(data_file, "w") as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('3,NA,178100\n')
    f.write('4,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2] # inputs取前两列，outputs取第三列
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