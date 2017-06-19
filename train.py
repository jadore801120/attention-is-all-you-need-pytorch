'''
This is a PyTorch implementation of "Attention is all you need".
'''

import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import Transformer
from DataLoader import DataLoader

def get_performance(crit, pred, gold, smoothing=False, num_class=None):
    ''' Apply label smoothing if needed '''

    # TODO: Add smoothing and PPL
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
        raise NotImplementedError

    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()

    return loss, n_correct


def train_epoch(model, training_data, crit, optimizer):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src, tgt)
        loss, n_correct = get_performance(crit, pred, gold)

        # backward
        loss.backward()
        optimizer.step()

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0] / len(training_data)

    return total_loss, n_total_correct/n_total_words

def eval_epoch(model, validation_data, crit):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):

        # prepare data
        src, tgt = batch
        gold = tgt[0][:, 1:]

        # forward
        pred = model(src, tgt)
        loss, n_correct = get_performance(crit, pred, gold)

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0] / len(validation_data)

    return total_loss, n_total_correct/n_total_words


def train(model, training_data, validation_data, crit, optimizer, opt):
    ''' Start training '''

    def update_learning_rate(n_steps):
        ''' Learning rate scheduling '''

        n_steps += 1
        new_lr = np.power(opt.d_model, -0.5) * np.min([
            np.power(n_steps, -0.5),
            np.power(opt.n_warmup_steps, -1.5) * n_steps])

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        update_learning_rate(epoch_i)
        train_loss, train_accu = train_epoch(model, training_data, crit, optimizer)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3} %'.format(
            loss=train_loss, accu=100*train_accu))

        valid_loss, valid_accu = eval_epoch(model, validation_data, crit)
        print('  - (Validation) loss: {loss: 8.5f}, accuracy: {accu:3.3} %'.format(
            loss=valid_loss, accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_word_vec', default=512)
    parser.add_argument('-d_model', default=512)
    parser.add_argument('-d_inner_hid', default=1024)
    parser.add_argument('-d_k', default=64)
    parser.add_argument('-d_v', default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.5)

    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    #========= Loading Dataset =========#
    data = torch.load(opt.data)

    #========= Preparing DataLoader =========#
    training_data = DataLoader(
        data['train']['src'],
        data['dict']['src'],
        data['train']['tgt'],
        data['dict']['tgt'],
        batch_size=opt.batch_size)

    validation_data = DataLoader(
        data['valid']['src'],
        data['dict']['src'],
        data['valid']['tgt'],
        data['dict']['tgt'],
        batch_size=opt.batch_size)

    #========= Preparing Model =========#
    transformer = Transformer(
        training_data.src_vocab_size,
        training_data.tgt_vocab_size,
        data['setting'].max_seq_len,
        proj_share_weight=True,
        embs_share_weight=True,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)

    print(transformer)

    optimizer = optim.Adam(
        transformer.get_trainable_parameters(),
        betas=(0.9, 0.98), eps=1e-09)

    def get_criterion(vocab_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(vocab_size)
        weight[Constants.PAD] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit = get_criterion(training_data.tgt_vocab_size)

    if opt.cuda:
        transformer = transformer.cuda()
        crit = crit.cuda()

    train(transformer, training_data, validation_data, crit, optimizer, opt)

if __name__ == '__main__':
    main()
