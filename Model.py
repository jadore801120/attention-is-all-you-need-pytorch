import torch
import torch.nn as nn
import torch.nn.init as init 
import numpy as np
import Constants
import torch.nn.functional as F

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, attn_mask=None):
        _, len_q, _ = q.size()
        _, len_k, _ = k.size()

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn.view(-1, len_k)).view(-1, len_q, len_k)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k=63, d_v=63, res_dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

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
        d_k = self.d_k
        d_v = self.d_v
        d_model = self.d_model
        for w_qi, w_ki, w_vi in zip(self.w_qs, self.w_ks, self.w_vs):
            q_i = w_qi(q.view(-1, d_model)).view((q.size(0), q.size(1), -1))
            k_i = w_ki(k.view(-1, d_model)).view((k.size(0), k.size(1), -1))
            v_i = w_vi(v.view(-1, d_model)).view((v.size(0), v.size(1), -1))
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

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head)
        self.enc_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=4, n_head=8, 
            d_word_vec=64, d_model=64, d_inner_hid=200, dropout=0.1,
            proj_share_weight=True, embs_share_weight=True):

        def position_encoding_init():
            ''' Init the sinusoid position encoding table '''

            d_pos_vec = d_word_vec
            n_pos = n_max_seq + 1

            # keep dim 0 for padding token position encoding zero vector
            position_enc = np.array([
                [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
                if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_pos)])

            position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
            position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1

            position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
            self.position_enc.weight.data = position_enc

        super(Transformer, self).__init__()

        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_max_seq+1, d_word_vec, padding_idx=Constants.PAD)
        position_encoding_init()

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

        if proj_share_weight:
            # Share the weight matrix between tgt word embedding/projection
            assert d_model == d_word_vec
            self.tgt_word_proj.weight = self.tgt_word_emb.weight

        if embs_share_weight:
            # Share the weight matrix between src/tgt word embeddings
            # assume the src/tgt word vec size are the same
            assert n_src_vocab == n_tgt_vocab
            self.src_word_emb.weight = self.tgt_word_emb.weight

        self.encode_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head) for _ in range(n_layers)])

        self.decode_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head) for _ in range(n_layers)])


    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freeze_param_ids = set(map(id, self.position_enc.parameters()))
        return (p for p in self.parameters() if id(p) not in freeze_param_ids)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        # Word embedding look up
        enc_input = self.src_word_emb(src_seq)
        dec_input = self.tgt_word_emb(tgt_seq)

        # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        dec_input += self.position_enc(tgt_pos)

        enc_outputs, dec_outputs = [], []
        enc_slf_attns, dec_slf_attns = [], []
        dec_enc_attns = []

        def get_attn_padding_mask(seq_q, seq_k):
            ''' Indicate the padding-related part to mask '''
            assert seq_q.dim() == 2 and seq_k.dim() == 2
            mb_size, len_k = seq_k.size()
            mb_size, len_q = seq_q.size()
            pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
            pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
            return pad_attn_mask

        # Encode
        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.encode_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            enc_outputs += [enc_output]
            enc_slf_attns += [enc_slf_attn]

        def get_attn_subsequent_mask(seq):
            ''' Get an attention mask to avoid using the subsequent info.'''
            assert seq.dim() == 2
            attn_shape = (seq.size(0), seq.size(1), seq.size(1))
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            subsequent_mask = torch.from_numpy(subsequent_mask)
            if seq.is_cuda:
                subsequent_mask = subsequent_mask.cuda()
            return subsequent_mask 

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(src_seq, tgt_seq)

        dec_output = dec_input
        for dec_layer, enc_output in zip(self.decode_stack, enc_outputs):
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=dec_slf_attn_mask, 
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            dec_outputs += [dec_output]
            dec_slf_attns += [dec_slf_attn]
            dec_enc_attns += [dec_enc_attn]

        # Final projection, the softmax will be done outside
        seq_logit = self.tgt_word_proj(dec_output.view(-1, dec_output.size(2)))

        return seq_logit
