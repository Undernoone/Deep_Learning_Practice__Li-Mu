import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l

def get_toekns_and_segments(tokens_a, tokens_b=None):
    tokens = ['[CLS]'] + tokens_a + ['[SEP]']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['[SEP]']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_layers,norm_shape, ffn_num_input, ffn_num_hiddens,
                 dropout, max_len=1000, key_size=768, query_size=768,num_heads):
        super(BERTEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"block{i}",
                d2l.EncoderBlock(key_size, query_size, num_hiddens,norm_shape, ffn_num_input, ffn_num_hiddens,
                                 dropout, num_heads))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments,valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X