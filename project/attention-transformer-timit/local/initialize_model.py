# a separate model initialization can split the parameter of model and trainning process,
# and help to improve the flexibility
# [to be done] should be able to convert kaldi nnet3 model into pytorch format?

import argparse
import kaldi_io
import torch
from transformer.Models import Transformer
from utils import instances_handler

def str2tuple(str):
    if str[0] == '(' and str[-1] == ')':
        arr = tuple(int(i) for i in str[1:-1].split(','))
        if len(arr) == 2:
            return arr
        else:
            print('[ERROR] invalid sub-sequence length!')
            exit(1)
    else:
        print('[ERROR] invalid sub-sequence string!')
        exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_feats_scp_file', required=True)
    parser.add_argument('-read_vocab_file', required=True)

    parser.add_argument('-encoder_max_len', type=int, required=True)
    parser.add_argument('-decoder_max_len', type=int, required=True)
    parser.add_argument('-src_fold', type=int, default=1)
    parser.add_argument('-encoder_sub_sequence', default='(-100,0)')
    parser.add_argument('-decoder_sub_sequence', default='(-20,0)')

    parser.add_argument('-en_layers', type=int, default=2)
    parser.add_argument('-de_layers', type=int, default=2)
    parser.add_argument('-n_head', type=int, default=3)
    parser.add_argument('-en_d_model', type=int, default=256)
    parser.add_argument('-de_d_model', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-save_model_file', required=True)
    opt = parser.parse_args()
    
    opt.encoder_sub_sequence = str2tuple(opt.encoder_sub_sequence)
    opt.decoder_sub_sequence = str2tuple(opt.decoder_sub_sequence)

    for key,matrix in kaldi_io.read_mat_scp(opt.read_feats_scp_file):
        opt.src_dim = matrix.shape[1]
        break
    print('[INFO] get feature of dimension {} from {}.'.format(opt.src_dim, opt.read_feats_scp_file))

    word2idx = instances_handler.read_vocab(opt.read_vocab_file)
    opt.tgt_vocab_dim = len(word2idx)
    print('[INFO] get label of dimension {} from {}.'.format(opt.tgt_vocab_dim, opt.read_vocab_file))

    print('[INFO] model will initialized with add_argument:\n\t{}.'.format(opt))

    model = Transformer(
        opt.src_dim,
        opt.tgt_vocab_dim,
        encoder_max_len=opt.encoder_max_len,
        decoder_max_len=opt.decoder_max_len,
        src_fold=opt.src_fold,
        encoder_sub_sequence=(-100,0),
        decoder_sub_sequence=(-20,0),
        en_layers=opt.en_layers,
        de_layers=opt.de_layers,
        n_head=opt.n_head,
        en_d_model=opt.en_d_model,
        de_d_model=opt.de_d_model,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout)

    checkpoint = {
        'model': model,
        'model_options': opt,
        'epoch': 0}

    torch.save(checkpoint, opt.save_model_file)
    #can be readed by:
    #checkpoint = torch.load(opt.save_model_file)
    #model = checkpoint['model']
    print('[INFO] initialized model is saved to {}.'.format(opt.save_model_file))


if __name__ == '__main__':
    main()