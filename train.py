"""This script handling the training process."""
import math
import time
import os

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


class OPT(object):
    def __init__(self):
        self.data = "data/multi30k.pkl"
        self.epoch = 10
        self.batch_size = 64
        self.d_model = 512
        self.d_inner_hid = 2048
        self.d_k = 64
        self.d_v = 64
        self.n_head = 8
        self.n_layers = 6
        self.n_warmup_steps = 4000
        self.dropout = 0.1
        self.embs_share_weight = False
        self.proj_share_weight = False
        self.log = "./log"
        self.save_model = True
        self.save_mode = 'best'  # choices=['all', 'best']
        self.no_cuda = True
        self.label_smoothing = False
        self.cuda = not self.no_cuda
        self.d_word_vec = self.d_model

        if not os.path.exists(self.log):
            os.makedirs(self.log)


def cal_performance(pred, gold, smoothing=False):
    """
    Apply label smoothing if needed
    :param pred:
    :param gold:
    :param smoothing:
    :return:
    """
    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    """
    Calculate cross entropy loss, apply label smoothing if needed.
    :param pred:
    :param gold:
    :param smoothing:
    :return:
    """
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    """
    Epoch operation in training phase
    :param model:
    :param training_data:
    :param optimizer:
    :param device:
    :param smoothing:
    :return:
    """
    model.train()  # nn.Model.train()将本层及子层的training设定为True # nn.Model.eval() # 将本层及子层的training设定为False
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    """
    Epoch operation in evaluation phase
    :param model:
    :param validation_data:
    :param device:
    :return:
    """
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    """
    Start training
    """
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = '{}/train.log'.format(opt.log)
        log_valid_file = '{}/valid.log'.format(opt.log)

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)  ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(train_loss, 100)),
            accu=100*train_accu,
            elapse=(time.time()-start)/60)
        )

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(valid_loss, 100)),
            accu=100*valid_accu,
            elapse=(time.time()-start)/60)
        )

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {'model': model_state_dict,
                      'settings': opt,
                      'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:

            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:

                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))

                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        dataset=TranslationDataset(src_word2idx=data['dict']['src'],
                                   tgt_word2idx=data['dict']['tgt'],
                                   src_insts=data['train']['src'],
                                   tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset=TranslationDataset(src_word2idx=data['dict']['src'],
                                   tgt_word2idx=data['dict']['tgt'],
                                   src_insts=data['valid']['src'],
                                   tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


def main():
    """
    Main function
    :return:
    """
    opt = OPT()
    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    # ========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        n_src_vocab=opt.src_vocab_size,
        n_tgt_vocab=opt.tgt_vocab_size,
        len_max_seq=opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98),
            eps=1e-09),
        opt.d_model,
        opt.n_warmup_steps
    )

    train(transformer, training_data, validation_data, optimizer, device, opt)


if __name__ == '__main__':
    main()
