''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer.Models import Transformer
from transformer.Beam import Beam

class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            proj_share_weight=model_opt.proj_share_weight,
            embs_share_weight=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner_hid=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)

        prob_projection = nn.LogSoftmax()

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        if opt.cuda:
            model.cuda()
            prob_projection.cuda()
        else:
            model.cpu()
            prob_projection.cpu()

        model.prob_projection = prob_projection

        self.model = model
        self.model.eval()

    def translate_batch(self, src_batch):
        ''' Translation work in one batch '''

        # Batch size is in different location depending on data.
        src_seq, src_pos = src_batch
        batch_size = src_seq.size(0)
        beam_size = self.opt.beam_size

        #- Enocde
        enc_outputs, enc_slf_attns = self.model.encoder(src_seq, src_pos)

        #--- Repeat data for beam
        src_seq = src_seq.unsqueeze(0).repeat(beam_size, 1, 1)\
        .transpose(0, 1).contiguous().view(batch_size * beam_size, 1, 1)
        enc_outputs = [
            Variable(enc_output.data.unsqueeze(0).repeat(beam_size, 1, 1, 1)\
            .transpose(0,1).contiguous().view(batch_size * beam_size, enc_output.size(1), -1)
            for enc_output in enc_outputs
        ]
        #--- Prepare beams
        beam = [Beam(beam_size, self.opt.cuda) for k in range(batch_size)]
        batch_idx = list(range(batch_size))
        n_remaining_sents = batch_size

        #- Decode
        for i in range(self.model_opt.max_token_seq_len):

            len_dec_seq = i + 1

            # -- Preparing decode data seq -- #
            input_data = torch.stack([
                b.get_current_state() for b in beam if not b.done]) # size: mb x bm x sq
            input_data = input_data.view(-1, len_dec_seq)           # size: (mb*bm) x sq
            input_data = Variable(input_data, volatile=True)

            # -- Preparing decode pos seq -- #
            # size: 1 x seq
            input_pos = torch.arange(1, len_dec_seq + 1).unsqueeze(0)
            # size: (batch * beam) x seq
            input_pos = input_pos.repeat(n_remaining_sents * beam_size, 1)
            input_pos = Variable(input_pos.type(torch.LongTensor), volatile=True)

            if self.opt.cuda:
                input_pos = input_pos.cuda()
                input_data = input_data.cuda()

            # -- Decoding -- #
            dec_outputs, dec_slf_attns, dec_enc_attns = self.model.decoder(
                input_data, input_pos, src_seq, enc_outputs)
            dec_output = dec_outputs[-1][:, -1, :] # (batch * beam) * d_model
            dec_output = self.model.tgt_word_proj(dec_output)
            out = self.model.prob_projection(dec_output)

            # batch x beam x n_words
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = self.tt.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active_enc_info(tensor_var, active_idx):
                ''' Remove the encoder outputs of finished instances in one batch. '''
                tensor_data = tensor_var.data.view(n_remaining_sents, -1, self.model_opt.d_model)

                new_size = list(tensor_var.size())
                new_size[0] = new_size[0] * len(active_idx) // n_remaining_sents

                # select the active index in batch
                return Variable(
                    tensor_data.index_select(0, active_idx).view(*new_size),
                    volatile=True)

            def update_active_seq(seq, active_idx):
                ''' Remove the src sequence of finished instances in one batch. '''
                view = seq.data.view(n_remaining_sents, -1)
                new_size = list(seq.size())
                new_size[0] = new_size[0] * len(active_idx) // n_remaining_sents # trim on batch dim

                # select the active index in batch
                return Variable(view.index_select(0, active_idx).view(*new_size), volatile=True)

            src_seq = update_active_seq(src_seq, active_idx)
            enc_outputs = [
                update_active_enc_info(enc_output, active_idx)
                for enc_output in enc_outputs]
            n_remaining_sents = len(active)

        #- Return useful information
        all_hyp, all_scores = [], []
        n_best = self.opt.n_best

        for b in range(batch_size):
            scores, ks = beam[b].sort_scores()
            all_scores += [scores[:n_best]]
            hyps = [beam[b].get_hypothesis(k) for k in ks[:n_best]]
            all_hyp += [hyps]

        return all_hyp, all_scores
