import re
import random
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 1. 读取和预处理你的数据集
def load_my_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 去掉特殊符号和空行
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', '', text)  # 保留中英文和数字
    return text

def preprocess(text):
    # 分词
    tokens = text.split()  # 对于中文需要用更合适的分词工具，如jieba
    # 构建词汇表
    vocab = d2l.Vocab(tokens)
    # 将文本转换为索引
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

# 加载并预处理数据
file_path = r'D:\MemoTrace\data\聊天记录\高子余(wxid_sboyih7p4vho22)\高子余_chat.txt'
text = load_my_data(file_path)
corpus, vocab = preprocess(text)

# 2. 创建数据迭代器
def data_iter_random(corpus, batch_size, num_steps):
    # 随机采样批量数据
    num_examples = (len(corpus) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [data(j * num_steps) for j in batch_indices]
        Y = [data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X), torch.tensor(Y)

# 设置batch_size和num_steps
batch_size, num_steps = 32, 35
train_iter = data_iter_random(corpus, batch_size, num_steps)


# 3. 定义GRU模型相关的函数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # update gate parameters
    W_xr, W_hr, b_r = three()  # reset gate parameters
    W_xh, W_hh, b_h = three()  # hidden gate parameters
    W_hq = normal((num_hiddens, num_outputs))  # output layer parameters
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        # @ 这个符号表示矩阵乘法
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)

# 4. 训练模型
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()
