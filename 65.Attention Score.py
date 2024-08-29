import math
import torch
from torch import nn
from d2l import torch as d2l

def mask_softmax(X, valid_lens):
    """ 如果没有给定有效长度(valid_lens)，直接在最后一个维度上执行softmax """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 如果有效长度是一维,则扩展
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 不是一维则展平成一维
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative (-1e6)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        # 对遮蔽后的张量恢复原来的形状并进行softmax运算
        return nn.functional.softmax(X.reshape(shape), dim=-1)

test_mask_softmax_init = torch.rand(2,2,4) # Two dim, Three rows, Four lines
print('test_mask_softmax:',test_mask_softmax_init)

# 第一个的只保留前两列，第二个的只保留前三列，后面的全被mask掉
print('test_mask_softmax:',mask_softmax(test_mask_softmax_init, torch.tensor([2,3])))
# 更细致的划分mask
print('test_mask_softmax:',mask_softmax(test_mask_softmax_init, torch.tensor([[1,3],[2,4]])))

# Markdown那几个公式
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 为了计算相似度得分，将查询扩展一个维度，使其形状与键相匹配
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = mask_softmax(scores, valid_lens)
        # values的shape: (batch_size, num_queries, 1, num_values, embed_size)
        # scores的shape: (batch_size, num_queries, num_keys)
        # attention_weights的shape: (batch_size, num_queries, num_keys)
        out = torch.bmm(self.dropout(self.attention_weights), values)
        return out.squeeze(1)


queries, keys = torch.normal(0, 1, (2,1,20)), torch.ones((2,10,2))
values = torch.arange(40, dtype=torch.float32).reshape(1,10,4).repeat(2,1,1)
valid_lens = torch.tensor([2,6])
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)

d2l.show_heatmaps(attention.attention_weights.reshape((1,1,2,10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()

class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = mask_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.normal(0,1,(2,1,2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)
d2l.show_heatmaps(attention.attention_weights.reshape((1,1,2,10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()