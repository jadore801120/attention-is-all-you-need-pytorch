''' Handling the data io '''
# import argparse
import torch
import transformer.Constants as Constants


def read_instances_from_file(inst_file, max_sent_len, keep_case):
    """
    将文件中的句子，统一小写，加入'<s>''</s>' 空格分词
    :param inst_file: 文件
    :param max_sent_len: 每句话的最大长度
    :param keep_case: false 对应 统一小写
    :return: 返回二维list
    """

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts


def build_vocab_idx(word_insts, min_word_count):
    """
    制作word2int
    :param word_insts:
    :param min_word_count:
    :return:
    """
    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {Constants.BOS_WORD: Constants.BOS,
                Constants.EOS_WORD: Constants.EOS,
                Constants.PAD_WORD: Constants.PAD,
                Constants.UNK_WORD: Constants.UNK}

    # 初始化 词-频 字典
    word_count = {w: 0 for w in full_vocab}
    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)  # 巧妙的编号方式
            else:
                ignored_word_count += 1
    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):
    """
    Mapping words to idx sequence.
    :param word_insts:
    :param word2idx:
    :return:
    """
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]


class OPT(object):
    def __init__(self):
        # self.train_src = "multi30k-dataset/data/task1/raw/train.en"
        # self.train_tgt = "multi30k-dataset/data/task1/raw/train.de"
        # self.valid_src = "multi30k-dataset/data/task1/raw/val.en"
        # self.valid_tgt = "multi30k-dataset/data/task1/raw/val.de"
        self.train_src = "data/multi30k/train.en"
        self.train_tgt = "data/multi30k/train.de"
        self.valid_src = "data/multi30k/val.en"
        self.valid_tgt = "data/multi30k/val.de"
        self.save_data = "data/multi30k.pkl"
        self.max_word_seq_len = 50
        self.min_word_count = 5
        self.keep_case = False
        self.share_vocab = False
        self.vocab = None


def main():
    # ''' Main function '''
    """
    原来的方法命令行形式传参数
    python preprocess.py
    -train_src data/multi30k/train.en.atok
    -train_tgt data/multi30k/train.de.atok
    -valid_src data/multi30k/val.en.atok
    -valid_tgt data/multi30k/val.de.atok
    -save_data data/multi30k.atok.low.pt
    :return:
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-train_src', required=True)  # data/multi30k/train.en.atok
    # parser.add_argument('-train_tgt', required=True)  # data/multi30k/train.de.atok
    # parser.add_argument('-valid_src', required=True)  # data/multi30k/val.en.atok
    # parser.add_argument('-valid_tgt', required=True)  # data/multi30k/val.de.atok
    # parser.add_argument('-save_data', required=True)  # data/multi30k.atok.low.pt
    # parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    # parser.add_argument('-min_word_count', type=int, default=5)
    # parser.add_argument('-keep_case', action='store_true')  # 命令行用keep_case时，keep_case为True，否则为False
    # parser.add_argument('-share_vocab', action='store_true')
    # parser.add_argument('-vocab', default=None)
    #
    # opt = parser.parse_args()

    opt = OPT()
    opt.max_token_seq_len = opt.max_word_seq_len + 2  # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tgt_word_insts = read_instances_from_file(opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    # Remove empty instances
    # list_tuple = zip(list1,list2) 把两个list合并成一个list，内部元素是元组
    # list1,list2 = list(zip(*list_tuple)) zip(* ) 相当于解压操作，在python3中返回一个对象，需要转换成list，才能拿到分别凯的list
    train_src_word_insts, train_tgt_word_insts = list(
        zip(*[(s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t])
    )

    # Validation set
    valid_src_word_insts = read_instances_from_file(opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    # Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(
        zip(*[(s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t])
    )

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data
        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {'settings': opt,
            'dict': {'src': src_word2idx,
                     'tgt': tgt_word2idx},
            'train': {'src': train_src_insts,
                      'tgt': train_tgt_insts},
            'valid': {'src': valid_src_insts,
                      'tgt': valid_tgt_insts}
            }

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')


if __name__ == '__main__':
    main()
