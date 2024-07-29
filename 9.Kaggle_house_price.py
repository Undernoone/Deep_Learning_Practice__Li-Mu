import hashlib
import os
import tarfile
import zipfile

import pandas as pd
import requests

Date_Hub = dict()
Data_Url = 'http://d21-data.s3-accelerate.amazonaws.com/'

def download(name,cache_dir=os.path.join('D:\\Deeplearning_Li Mu_Date\\Kaggle_House_Price\\data','data')):
    assert name in Date_Hub, f"{name} 不存在于 {Date_Hub}."
    url , sha1_hash = Date_Hub[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname,'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url,stream=True,verify=True)
    with open(fname,'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp,extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_UHB中的所有文件"""
    for name in Date_Hub:
        download(name)

Date_Hub['kaggle_house_train'] = (Data_Url + 'kaggle_house_pred_train.csv','585e9cc9370b9160e7921475fbcd7d31219ce')
Date_Hub['kaggle_house_test'] = (Data_Url + 'kaggle_house_pred_test.csv', 'fal9780a7b011d9b009e8bff8e99922a8ee2eb90')
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
print(train_data.shape) # 1460个样本，80个te特征，1个标号label
print(test_data.shape) # 测试样本没有标号label
print(train_data.shape)  # 检查数据形状
print(train_data.columns)  # 打印所有列名

print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]]) # 前面四行的某些列特征