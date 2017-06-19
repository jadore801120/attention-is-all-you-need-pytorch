''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import transformer.Constants as Constants

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self,
            src_insts, src_word2idx,
            tgt_insts, tgt_word2idx,
            cuda=True, batch_size=64):

        assert len(src_insts) == len(tgt_insts)
        assert len(src_insts) >= batch_size

        self.cuda = cuda
        self._n_batch = (len(src_insts) // batch_size) - 1
        self._batch_size = batch_size

        self._src_insts = src_insts
        self._tgt_insts = tgt_insts

        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}

        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word

        self._iter_count = 0
        self.shuffle()

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

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        paired_insts = list(zip(self._src_insts, self._tgt_insts))
        random.shuffle(paired_insts)
        self._src_insts, self._tgt_insts = zip(*paired_insts)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch


    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])

            inst_data_tensor = Variable(torch.LongTensor(inst_data))
            inst_position_tensor = Variable(torch.LongTensor(inst_position))

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            self._iter_count += 1

            start_idx = self._iter_count * self._batch_size
            end_idx = (self._iter_count + 1) * self._batch_size

            src_insts = self._src_insts[start_idx:end_idx]
            tgt_insts = self._tgt_insts[start_idx:end_idx]

            src_data, src_pos = pad_to_longest(src_insts)
            tgt_data, tgt_pos = pad_to_longest(tgt_insts)

            return (src_data, src_pos), (tgt_data, tgt_pos)
        else:
            self.shuffle()
            self._iter_count = 0
            raise StopIteration()
