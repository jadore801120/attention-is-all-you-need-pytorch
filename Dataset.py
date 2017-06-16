''' Handling the data io '''

import random
import Constants
import numpy as np
import torch
from torch.autograd import Variable

def build_dataset(
        pos_inst_file,
        neg_inst_file,
        opt,
        batch_first=False,
        return_length=True,
        include_sentence_symbols=False,
        word2idx=None):
    ''' Do the preprocessing and create a data loader '''

    def read_word_sequences_from_file(instance_file):
        '''Convert file into word lists and vocab'''

        print("Read instances from file: {}".format(instance_file))

        def truncate_to_maximum_length(seq):
            ''' Truncate instances to the specified size '''
            return [w for i, w in enumerate(seq) if i < opt.max_seq_len]

        _w_insts = []
        with open(instance_file) as f:
            for i, sent in enumerate(f):
                words = sent.split()
                _w_insts += [truncate_to_maximum_length(words)]

        if include_sentence_symbols:
            for i, _w_inst_i in enumerate(_w_insts):
                _w_insts[i] = [Constants.BOS_WORD] + _w_inst_i + [Constants.EOS_WORD]

        _vocab = set(w for com in _w_insts for w in com)

        return _w_insts, _vocab

    def build_vocab_idx(w_insts, full_vocab):
        '''Trim vocab by number of occurence'''
        _word_count = {w: 0 for w in full_vocab}

        for sent in w_insts:
            for word in sent:
                _word_count[word] += 1

        _word2idx = {
            Constants.BOS_WORD: Constants.BOS,
            Constants.EOS_WORD: Constants.EOS,
            Constants.PAD_WORD: Constants.PAD,
            Constants.UNK_WORD: Constants.UNK}

        symbols = {
            Constants.BOS_WORD,
            Constants.EOS_WORD,
            Constants.PAD_WORD,
            Constants.UNK_WORD}

        ignored_word_count = 0
        for word, count in _word_count.items():
            if word in symbols or count > opt.min_word_count:
                _word2idx[word] = len(_word2idx)
            else:
                ignored_word_count += 1

        print(
            'Trimmed vocabulary size = {}, with word minimum occurrence = {}'
            .format(len(_word2idx), opt.min_word_count))

        print("Ignored word count = {}".format(ignored_word_count))

        return _word2idx

    def convert_instance_to_idx_seq(w_insts, word2idx):
        '''Word mapping to idx'''
        return [
            [word2idx[w] if w in word2idx else Constants.UNK for w in s]
            for s in w_insts]

    _pos_w_insts, _pos_vocab = read_word_sequences_from_file(pos_inst_file)
    _neg_w_insts, _neg_vocab = read_word_sequences_from_file(neg_inst_file)

    print('Positive instance count = {}'.format(len(_pos_w_insts)))
    print('Negative instance count = {}'.format(len(_neg_w_insts)))

    _w_insts = _pos_w_insts + _neg_w_insts
    _full_vocab = _neg_vocab | _pos_vocab
    print('Full vocabulary size = {}'.format(len(_full_vocab)))

    if not word2idx:
        _word2idx = build_vocab_idx(_w_insts, _full_vocab)
    else:
        print("Predefined vocabulary given.")
        _word2idx = word2idx
        print("Ignored word count = {}".format(
            len([w for w in _full_vocab if w not in _word2idx])))

    _pos_idx_insts = convert_instance_to_idx_seq(_pos_w_insts, _word2idx)
    _neg_idx_insts = convert_instance_to_idx_seq(_neg_w_insts, _word2idx)

    _idx_insts = _pos_idx_insts + _neg_idx_insts
    _labels = [1] * len(_pos_idx_insts) + [0] * len(_neg_idx_insts)

    return DataLoader(
        _idx_insts,
        _labels,
        _word2idx,
        batch_size=opt.batch_size,
        batch_first=batch_first,
        return_length=return_length,
        cuda=opt.cuda)

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self,
            instances,
            labels,
            word2idx,
            batch_size=20,
            return_length=True,
            return_position=True,
            batch_first=False,
            cuda=True):

        assert len(instances) == len(labels)
        assert len(instances) >= batch_size

        self.cuda = cuda
        self.labels = labels
        self.instances = instances
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.return_length = return_length
        self.return_position = return_position

        self._word2idx = word2idx
        self._idx2word = {idx:w for w, idx in self._word2idx.items()}
        self._vocab_size = len(word2idx)

        self._batch_count = len(labels) // batch_size
        self._iter_count = 0
        self.shuffle()

    @property
    def vocab_size(self):
        ''' Attribute for vocab size '''
        return self._vocab_size

    @property
    def word2idx(self):
        ''' Attribute for word dictionary '''
        return self._word2idx

    @property
    def idx2word(self):
        ''' Attribute for index dictionary '''
        return self._idx2word

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def _pad_to_longest(self, batch_inst, batch_label):

        if self.return_length:
            batch_data = list(zip(batch_inst, batch_label))
            batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)
            batch_inst, batch_label = zip(*batch_data)

        batch_lens = [len(inst) for inst in batch_inst]
        max_data_len = max(batch_lens)

        batch_pad_data = np.array([
            inst + [Constants.PAD] * (max_data_len-len(inst))
            for inst in batch_inst])

        if not self.batch_first:
            batch_pad_data = batch_pad_data.transpose()

        batch_data_tensor = Variable(torch.LongTensor(batch_pad_data).contiguous())
        batch_label = Variable(torch.LongTensor(batch_label).contiguous())
        batch_lens = Variable(torch.LongTensor(batch_lens))


        batch_position = np.array([
            [i if w_i != Constants.PAD else 0 for i, w_i in enumerate(inst)]
            for inst in batch_pad_data])
        batch_poss = Variable(torch.LongTensor(batch_position))
        if self.cuda:
            batch_data_tensor = batch_data_tensor.cuda()
            batch_label = batch_label.cuda()
            batch_lens = batch_lens.cuda()
            batch_poss = batch_poss.cuda()
        if self.return_position:
            batch_inst = (batch_data_tensor, batch_poss)

        return batch_inst, batch_label

    def next(self):
        ''' Python 2 compatibility '''

        if self._iter_count < self.batch_count:
            start_idx = self._iter_count*self.batch_size
            end_idx = (self._iter_count+1)*self.batch_size

            batch_inst = self.instances[start_idx:end_idx]
            batch_label = self.labels[start_idx:end_idx]

            batch_inst, batch_label = self._pad_to_longest(batch_inst, batch_label)
            self._iter_count += 1
            return (batch_inst, batch_label)
        else:
            self._iter_count = 0
            self.shuffle()
            raise StopIteration()

    @property
    def batch_count(self):
        ''' Get the total count '''
        return self._batch_count

    def __len__(self):
        return self.batch_count

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        data = list(zip(self.instances, self.labels))
        random.shuffle(data)
        self.instances, self.labels = zip(*data)
