''' Define the sublayers in encoder/decoder layer '''

import torch
import torch.nn as nn
from transformer.Modules import Linear, ScaledDotProductAttention, LayerNormalization

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k=64, d_v=64, res_dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.w_qs = nn.ModuleList([
            Linear(d_model, d_k, bias=False) for _ in range(n_head)])
        self.w_ks = nn.ModuleList([
            Linear(d_model, d_k, bias=False) for _ in range(n_head)])
        self.w_vs = nn.ModuleList([
            Linear(d_model, d_v, bias=False) for _ in range(n_head)])

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)
        self.dropout = nn.Dropout(res_dropout)

    def forward(self, q, k, v, attn_mask=None):
        residual = q

        outputs, attns = [], []
        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        for w_qi, w_ki, w_vi in zip(self.w_qs, self.w_ks, self.w_vs):
            q_i = w_qi(q.view(-1, d_model)).view((mb_size, len_q, -1))
            k_i = w_ki(k.view(-1, d_model)).view((mb_size, len_k, -1))
            v_i = w_vi(v.view(-1, d_model)).view((mb_size, len_v, -1))
            output, attn = self.attention(q_i, k_i, v_i, attn_mask=attn_mask)
            outputs += [output]
            attns += [attn]

        outputs = torch.cat(outputs, 2)
        outputs = self.proj(outputs.view(-1, outputs.size(2))).view_as(residual)
        outputs = self.dropout(outputs)
        attns = torch.cat(attns)

        return self.layer_norm(outputs + residual), attns

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, res_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(res_dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)
