'''
This is a PyTorch implementation of Attention is all you need.
'''

import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import Constants
import Dataset
from Model import Transformer

def train(transformer, training_data, crit, optimizer, epoch):
    ''' Game's on '''

    for epoch_i in range(epoch):
        print('[Epoch {}]'.format(epoch_i))
        pbar = tqdm(
            training_data, mininterval=2,
            desc='Epoch {} '.format(epoch_i))

        total_loss = 0
        for batch in pbar:
            optimizer.zero_grad()

            batch_inst, _ = batch

            src, tgt = batch_inst, batch_inst

            optimizer.zero_grad()
            pred = transformer(src, (tgt[0][:, :-1], tgt[1][:, :-1]))
            gold = tgt[0][:, 1:]
            loss = get_loss(
                    crit, pred, gold, 
                    training_data.vocab_size, smoothing=False)
            loss.backward()
            optimizer.step()
            total_loss += loss.data[0]
        print(total_loss/len(training_data))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('train_pos_inst_file')
    parser.add_argument('train_neg_inst_file')

    parser.add_argument('-epoch', type=int, default=10)

    parser.add_argument('-max_seq_len', type=int, default=20,
                        help='maximum sequence length')
    parser.add_argument('-min_word_count', type=int, default=5,
                        help='Word embedding sizes')

    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-z_size', type=int, default=500)
    parser.add_argument('-num_class', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-max_norm', type=float, default=3.0)
    parser.add_argument('-lr', type=float, default=1e-3)

    parser.add_argument('-rnn_size', default=300)
    parser.add_argument('-word_vec_size', default=300)
    parser.add_argument('-z_dim', default=50)
    parser.add_argument('-max_sent_length', default=12)

    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    #========= Preparing Dataset =========#
    training_data = Dataset.build_dataset(
        opt.train_pos_inst_file,
        opt.train_neg_inst_file,
        opt,
        include_sentence_symbols=True,
        using_context=False,
        return_length=False,
        batch_first=True)

    #========= Preparing Model =========#
    transformer = Transformer(training_data.vocab_size, n_layers=1)
    crit = get_criterion(training_data.vocab_size)
    optimizer = optim.Adam(
        transformer.get_trainable_parameters(),
        betas=(0.9, 0.98), eps=1e-09)

    if opt.cuda:
        transformer = transformer.cuda()
        crit = crit.cuda()

    train(transformer, training_data, crit, optimizer, opt.epoch)

def get_criterion(vocab_size):
    ''' With PAD token zero weight '''
    weight = torch.ones(vocab_size)
    weight[Constants.PAD] = 0
    return nn.CrossEntropyLoss(weight, size_average=False)

def get_loss(crit, pred, gold, num_class, smoothing=False):
    ''' Apply label smoothing if needed '''
    gold = Variable(gold.data, requires_grad=False)
    if smoothing:
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
    return crit(pred, gold.contiguous().view(-1))

def update_learning_rate(optimizer, d_model, n_steps):
    ''' 1. Increasing the learning rate
           linearly for the first n_warmup_steps training steps.
        2. Decreasing it thereafter proportionally
           to the inverse square root of the step number. '''

    n_warmup_steps = 4000
    lr = np.power(d_model, -0.5) * np.min(
        np.power(n_steps, -0.5),
        np.power(n_warmup_steps, -1.5) * n_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
