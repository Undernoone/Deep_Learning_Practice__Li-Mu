import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(shape, device=device) * 0.01
    # sup function to create three sets of parameters
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    '''
    写段注释，方便看，我的大脑内存不是很大，笨就写在旁边方便自己看。
    W_xz：从输入到更新门的权重，相当于在地铁类比中，前一时刻的隐藏状态和当前输入信息在2号线上的传递。
    W_hz：从隐藏状态到更新门的权重，决定了前一时刻的隐藏状态如何影响当前的更新。
    b_z：更新门的偏置项，用于调整更新门的输出。
    W_xr：从输入到重置门的权重，相当于在地铁类比中，决定前一时刻隐藏状态和当前输入信息在2号线上的传递情况。
    W_hr：从隐藏状态到重置门的权重，决定前一时刻的隐藏状态如何被“重置”。
    b_r：重置门的偏置项，用于调整重置门的输出。
    W_xh：从输入到候选隐藏状态的权重，相当于重置门处理后的信息换乘4号线并生成候选信息。
    W_hh：从隐藏状态到候选隐藏状态的权重，影响候选信息的生成。
    b_h：生成候选隐藏状态的偏置项，用于调整候选信息的输出。
    W_hq：从隐藏状态到输出的权重，相当于更新后的信息在4号线上的传递，最终生成当前的隐藏状态输出。
    b_q：输出的偏置项，用于调整输出的最终结果。
    '''
    W_xz, W_hz, b_z = three()
    W_xr, W_hr, b_r = three()
    W_xh, W_hh, b_h = three()
    W_hq = normal((num_hiddens, num_outputs))
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
        # @ 这个符号表示矩阵乘法，很多写法
        # Z-更新门 、 R-重置门 、 H_tilda-候选隐藏状态 、 H-隐藏状态 、 Y-输出
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()

# PyTorch 实现
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()