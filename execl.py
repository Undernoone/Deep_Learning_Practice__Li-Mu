
import pandas as pd
import os
# 数据预处理
os.makedirs(os.path.join("..", "Deeplearning_Li Mu_Date"), exist_ok=True) # 创建文件夹
data_file = os.path.join("..", "Deeplearning_Li Mu_Date", "data.csv") # 文件路径

with open(data_file, "w") as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('3,NA,178100\n')
    f.write('4,NA,140000\n')