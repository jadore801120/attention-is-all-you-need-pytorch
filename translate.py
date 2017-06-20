import torch
import argparse
from tqdm import tqdm
from transformer.Translator import Translator
from DataLoader import DataLoader
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-vocab', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_seq_len', type=int, default=20,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-no_cuda', action='store_true')

opt = parser.parse_args()
opt.cuda = not opt.no_cuda

def main():

    predefined_data = torch.load(opt.vocab)
    test_src_word_insts = read_instances_from_file(opt.src, opt.max_seq_len, False)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, predefined_data['dict']['src'])

    test_data = DataLoader(
        predefined_data['dict']['src'],
        predefined_data['dict']['tgt'],
        src_insts=test_src_insts,
        batch_size=opt.batch_size)

    translator = Translator(opt)
    translator.model.eval()

    for batch in tqdm(test_data, mininterval=2, desc='  - (Test)', leave=False):
        all_hyp, all_scores, gold_scores = translator.translate_batch(batch)
        for idx_seqs in all_hyp:
            for idx_seq in idx_seqs: 
                print([test_data.tgt_idx2word[idx] for idx in idx_seq])

if __name__ == "__main__":
    main()
