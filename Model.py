import torch
import torch.nn as nn
import numpy as np
import Constants

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        #print('ln out', ln_out.size())
        #print('z', z.size())
        #print('a', self.a_2.size())
        #print('b', self.b_2.size())
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
        #print('q', q.size())
        #print('k', k.size())
        #print('v', v.size())
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        #print('attn', attn.size())
        attn = self.dropout(attn).view(-1, k.size(1))
        attn = self.softmax(attn).view(q.size(0), -1, k.size(1))
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -float('inf'))
        output = torch.bmm(attn, v)
        #print('output', output.size())
        
        return attn, output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_hids=(64, 64, 64), res_dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        d_q, d_w, d_v = d_hids
        self.d_q = d_q # debug
        self.w_Qs = nn.ModuleList([
            nn.Linear(d_model, d_q, bias=False) for _ in range(n_head)])
        self.w_Ks = nn.ModuleList([
            nn.Linear(d_model, d_w, bias=False) for _ in range(n_head)])
        self.w_Vs = nn.ModuleList([
            nn.Linear(d_model, d_v, bias=False) for _ in range(n_head)])

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.reshape_proj = nn.Linear(n_head*d_model, d_model)
        self.dropout = nn.Dropout(res_dropout)

    def forward(self, q, k, v):
        residual = q
        d_hid = self.d_q # debug
        '''
        attns, outputs = list(zip(*[
            self.attention(
                #w_Qi(q), w_Ki(k), w_Vi(v))
                w_Qi(q.view(-1, d_hid)), w_Ki(k.view(-1, d_hid)), w_Vi(v.view(-1, d_hid)))
            for w_Qi, w_Ki, w_Vi in zip(
                self.w_Qs, self.w_Ks, self.w_Vs)]))
        '''
        attns = []
        outputs = []
        for w_Qi, w_Ki, w_Vi in zip(self.w_Qs, self.w_Ks, self.w_Vs):
            q = w_Qi(q.view(-1, d_hid)).view_as(q) 
            k = w_Ki(k.view(-1, d_hid)).view_as(k) 
            v = w_Vi(v.view(-1, d_hid)).view_as(v)
            attn, output = self.attention(q, k, v) 
            attns += [attn]
            outputs += [output]

        outputs = torch.cat(outputs, 2)
        outputs = self.reshape_proj(outputs.view(-1, outputs.size(2)))
        outputs = self.dropout(outputs.view_as(residual))
        attns = torch.cat(attns)
        #print(outputs.size())
        #print(residual.size())

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
        output = self.w_2(output).transpose(1, 2)
        return self.layer_norm(self.dropout(output) + residual)

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner_hid, n_head):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_head)
        self.enc_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid)

    def forward(self, dec_input, enc_output):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_vocab, n_layers=4, n_head=8, n_max_seq=50,
            d_word_vec=64, d_model=64, d_inner_hid=1024,
            dropout=0.1, share_weight=True):

        super(Transformer, self).__init__()
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_max_seq+1, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.apply(self._position_encoding_init)

        self.src_word_emb = nn.Embedding(
            n_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.tgt_word_emb = nn.Embedding(
            n_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.word_proj = nn.Linear(d_model, n_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

        '''
        if share_weight:
            # Share the weight matrix
            assert d_model == d_word_vec
            self.word_proj.weight = self.src_word_emb.weight = self.tgt_word_emb.weight
        '''

        self.encode_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head) for _ in range(n_layers)])

        self.decode_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head) for _ in range(n_layers)])

    def _position_encoding_init(self, module):
        ''' Init the fixed sinusoid position encoding table '''

        classname = module.__class__.__name__
        if classname.find('Embedding') != -1:

            d_pos_vec = self.d_model
            n_pos = self.n_max_seq + 1

            # keep dim 0 for padding token position encoding zero vector
            position_enc = np.array([
                [pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
                if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_pos)])

            position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
            position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1

            module.weight.data = torch.from_numpy(position_enc).type(torch.FloatTensor)

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freeze_param_ids = set(map(id, self.position_enc.parameters()))
        return (p for p in self.parameters() if id(p) not in freeze_param_ids)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt
        #print(src_seq.size(), tgt_seq.size())

        # Word embedding look up
        enc_input = self.src_word_emb(src_seq)
        dec_input = self.tgt_word_emb(tgt_seq)

        # Position Encoding addition
        enc_input += self.position_enc(src_pos)
        dec_input += self.position_enc(tgt_pos)

        enc_outputs, dec_outputs = [], []
        enc_slf_attns, dec_slf_attns = [], []
        dec_enc_attns = []

        # Encode
        #attn_mask = src_seq.data.eq(Constants.PAD) # padding mask
        enc_output = enc_input
        for enc_layer in self.encode_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_outputs += [enc_output]
            enc_slf_attns += [enc_slf_attn]

        # Decode

        '''
        padding_mask = tgt_seq.data.eq(Constants.PAD)
        subsequent_mask = torch.from_numpy(
            np.triu(np.ones([2, 3, 3]).astype('uint8'), k=1))
        attn_mask = torch.gt(subsequent_mask + padding_mask, 0)
        '''

        dec_output = dec_input
        for dec_layer, enc_output in zip(self.decode_stack, enc_outputs):
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output)
            dec_outputs += [dec_output]
            dec_slf_attns += [dec_slf_attn]
            dec_enc_attns += [dec_enc_attn]

        # Final projection, the softmax will be done outside
        seq_logit = self.word_proj(dec_output.view(-1, dec_output.size(2)))

        return seq_logit
