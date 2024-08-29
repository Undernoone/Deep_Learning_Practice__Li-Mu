import torch
from torch import nn
from d2l import torch as d2l

# Traditional LSTM
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal(
            (num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
    '''
    W_xi：从输入到输入门的权重，相当于地铁类比中，前一时刻的隐藏状态和当前输入信息在1号线上的传递，决定输入信息进入输入门的程度。
    W_hi：从隐藏状态到输入门的权重，决定前一时刻的隐藏状态如何影响输入门，相当于前一站（隐藏状态）乘客换乘1号线到输入门的通道。
    b_i：输入门的偏置项，用于调整输入门的输出，相当于调整1号线的通行规则。
    W_xf：从输入到遗忘门的权重，相当于地铁类比中，前一时刻的隐藏状态和当前输入信息在1号线上的传递，决定信息如何进入遗忘门。
    W_hf：从隐藏状态到遗忘门的权重，决定前一时刻的隐藏状态如何影响遗忘门，相当于前一站（隐藏状态）乘客换乘1号线到遗忘门的通道。
    b_f：遗忘门的偏置项，用于调整遗忘门的输出，相当于调整1号线的通行规则。
    W_xo：从输入到输出门的权重，相当于地铁类比中，前一时刻的隐藏状态和当前输入信息在1号线上的传递，决定信息如何进入输出门。
    W_ho：从隐藏状态到输出门的权重，决定前一时刻的隐藏状态如何影响输出门，相当于前一站（隐藏状态）乘客换乘1号线到输出门的通道。
    b_o：输出门的偏置项，用于调整输出门的输出，相当于调整1号线的通行规则。
    W_xc：从输入到候选记忆细胞的权重，相当于地铁类比中，当前输入信息经过1号线处理后生成候选记忆细胞的通道。
    W_hc：从隐藏状态到候选记忆细胞的权重，影响候选记忆细胞的生成，相当于前一站（隐藏状态）乘客经过1号线生成候选记忆细胞的通道。
    b_c：候选记忆细胞的偏置项，用于调整候选记忆细胞的输出，相当于调整生成候选记忆细胞的通行规则。
    W_hq：从隐藏状态到输出的权重，相当于在3号线上的传递，将更新后的隐藏状态生成最终的输出结果。
    b_q：输出的偏置项，用于调整最终输出的结果，相当于调整3号线的通行规则。
    '''
    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()
    W_hq = normal((num_hiddens, num_outputs))
    b_q  = torch.zeros(num_outputs, device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]

    for param in params:
        param.requires_grad_(True)

    return params

def init_lstm_state(batch_size, num_hiddens, device):
    # 返回一个元组，包含两个张量：一个全零张量表示初始的隐藏状态，和一个全零张量表示初始的记忆细胞状态。
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens),device=device))

def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        # I - 输入门、F - 遗忘门、O - 输出门、C_tilda - 候选记忆细胞、C - 记忆细胞、H - 隐藏状态、Y - 输出
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()

# Pytorch LSTM
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
mode = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()