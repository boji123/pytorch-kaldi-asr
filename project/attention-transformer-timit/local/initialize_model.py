# a separate model initialization can split the parameter of model and trainning process,
# and help to improve the flexibility
# [to be done] should be able to convert kaldi nnet3 model into pytorch format?

import argparse
import kaldi_io
import torch
from transformer.Models import Transformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_feats_scp_file', required=True)
    parser.add_argument('-read_vocab_file', required=True)
    parser.add_argument('-max_token_seq_len', type=int, required=True)

    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-embs_share_weight', action='store_true')

    parser.add_argument('-save_model_file', required=True)
    opt = parser.parse_args()

    print('--------------------[PROCEDURE]--------------------')
    print('[PROCEDURE] reading dimension from data file and initialize the model')

    for key,matrix in kaldi_io.read_mat_scp(opt.read_feats_scp_file):
        opt.src_dim = matrix.shape[1]
        break
    print('[INFO] get feature of dimension {} from {}.'.format(opt.src_dim, opt.read_feats_scp_file))

    word2idx = torch.load(opt.read_vocab_file)
    opt.tgt_vocab_dim = len(word2idx)
    print('[INFO] get label of dimension {} from {}.'.format(opt.tgt_vocab_dim, opt.read_vocab_file))

    print('[INFO] model will initialized with add_argument:\n{}.'.format(opt))

    model = Transformer(
        opt.src_dim,
        opt.tgt_vocab_dim,
        opt.max_token_seq_len,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_model=opt.d_model,
        d_inner_hid=opt.d_inner_hid,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        proj_share_weight=opt.proj_share_weight,
        embs_share_weight=opt.embs_share_weight)

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