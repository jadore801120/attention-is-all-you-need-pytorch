''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants

parser = argparse.ArgumentParser()
parser.add_argument('-train_src', required=True)
parser.add_argument('-train_tgt', required=True)
parser.add_argument('-valid_src', required=True)
parser.add_argument('-valid_tgt', required=True)
parser.add_argument('-output', required=True)
parser.add_argument('-max_seq_len', type=int, default=20)
parser.add_argument('-min_word_count', type=int, default=5)
parser.add_argument('-keep_case', action='store_true')
parser.add_argument('-share_vocab', action='store_true')

opt = parser.parse_args()

def read_instances_from_file(inst_file):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    with open(inst_file) as f:
        for i, sent in enumerate(f):
            if not opt.keep_case:
                sent = sent.lower()
            words = sent.split()
            word_inst = words[:opt.max_seq_len]
            word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))
    return word_insts

def build_vocab_idx(word_insts):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > opt.min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(opt.min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    '''Word mapping to idx'''
    return [[word2idx[w] if w in word2idx else Constants.UNK for w in s] for s in word_insts]

def main():
    ''' Main function '''

    train_src_word_insts = read_instances_from_file(opt.train_src)
    train_tgt_word_insts = read_instances_from_file(opt.train_tgt)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    valid_src_word_insts = read_instances_from_file(opt.valid_src)
    valid_tgt_word_insts = read_instances_from_file(opt.valid_tgt)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    if opt.share_vocab:
        print('[Info] Build shared vocabulary for source and target.')
        word2idx = build_vocab_idx(train_src_word_insts + train_tgt_word_insts)
        src_word2idx = tgt_word2idx = word2idx
    else:
        print('[Info] Build vocabulary for source.')
        src_word2idx = build_vocab_idx(train_src_word_insts)
        print('[Info] Build vocabulary for target.')
        tgt_word2idx = build_vocab_idx(train_tgt_word_insts)

    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'setting': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_tgt_insts,
            'tgt': valid_src_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.output)
    torch.save(data, opt.output)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
