import numpy as np
import torch
import torch.utils.data

from transformer import Constants

def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self, src_word2idx, tgt_word2idx,
        src_insts=None, tgt_insts=None):

        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts:
            return self._src_insts[idx], self._tgt_insts[idx]
        return self._src_insts[idx]
