''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import transformer.Constants as Constants

class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, src_word2idx, tgt_word2idx, src_insts=None, tgt_insts=None, cuda=True, batch_size=64, drop_last=False, mode='train'):
        assert src_insts
        assert len(src_insts) >= batch_size

        if tgt_insts:
            assert len(src_insts) == len(tgt_insts)

        self.cuda = cuda
        self.mode = mode
        self.drop_last = drop_last

        self.total_samples = len(src_insts)
        if self.drop_last: 
            self._n_batch = len(src_insts) // self.batch_size
        else:
            self._n_batch = (len(src_insts) + batch_size - 1) // batch_size

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
        # for test data, we should not shuffle, because it is one to one map 
        if mode == 'train':
            self.shuffle()
        else: 
            pass # we donot shuffle when it is testing or validation set

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

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._tgt_insts: # if there exists _tgt_insts 
            paired_insts = list(zip(self._src_insts, self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_insts, self._tgt_insts = zip(*paired_insts)
        else:
            random.shuffle(self._src_insts)


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

            inst_data = np.array([inst + [Constants.PAD] * (max_len - len(inst)) for inst in insts])
            # 1, 2, ..., 0, 0, ..0 (0 means pad position) for each instance
            inst_position = np.array([[pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)] 
                for inst in inst_data])

            inst_data_tensor = Variable(torch.LongTensor(inst_data))
            inst_position_tensor = Variable(torch.LongTensor(inst_position))

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch: # can be self._n_batch-1(33)

            start_idx = self._iter_count * self._batch_size # 0 * 30, ..., 33 * 30 

            self._iter_count = self._iter_count + 1

            end_idx = self._iter_count * self._batch_size # 34 * 30

            # last batch, if we donot drop_last 
            if (not self.drop_last) and end_idx > self.total_samples:
                end_idx = self.total_samples # note slicing doesnot include last index

            src_insts = self._src_insts[start_idx:end_idx]
            # pad the data
            src_data, src_pos = pad_to_longest(src_insts)

            if not self._tgt_insts:
                return src_data, src_pos # return 
            else: # if contains target
                tgt_insts = self._tgt_insts[start_idx:end_idx]
                tgt_data, tgt_pos = pad_to_longest(tgt_insts)
                return (src_data, src_pos), (tgt_data, tgt_pos)

        else: # start a new epoch of loading data
            if self.mode == 'train': # maybe we should add another flag to control whether to shuffle it 
                self.shuffle()
            else:
                pass 

            self._iter_count = 0
            raise StopIteration()
