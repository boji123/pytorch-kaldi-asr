import argparse
import time
from tqdm import tqdm
import numpy as np

from utils import instances_handler
from utils.BatchLoader import BatchLoader
from utils import constants

import torch
import torch.nn as nn

from torch.autograd import Variable
from transformer.Models import Transformer
from transformer.Models import fold_seq_and_mask
from transformer.Lattice import Lattice

from train import initialize_batch_loader

from utils.get_gpu import get_available_gpu_ids

def translate_batch(model, batch, opt, model_options):
    model.eval()
    # prepare data
    src_seq = batch[1]
    src_pad_mask = batch[2]
    tgt_seq = batch[3]

    src_seq = torch.FloatTensor(src_seq) #batch * max length in batch * padded feature dim
    src_pad_mask = torch.ByteTensor(src_pad_mask) #batch * maxlength in batch * bool mask dim
    # warning: embedding require long tensor, maybe it's a waste of menory, waiting to be solved
    tgt_seq = torch.LongTensor(tgt_seq) #batch * max length in batch * padded index dim

    if opt.use_gpu:
        src_seq = src_seq.cuda()
        src_pad_mask = src_pad_mask.cuda()
        tgt_seq = tgt_seq.cuda()

    goal = tgt_seq[:, 1:]
    tgt_seq = tgt_seq[:, :-1]

    beam_size = opt.beam_size
    batch_size = src_seq.size(0)
    #---------------------------------------------------------------------------------------
    #- Enocde
    src_seq, src_pad_mask = fold_seq_and_mask(src_seq, src_pad_mask, model.src_fold)
    enc_output, *_ = model.encoder(src_seq, src_pad_mask)

    #--- Prepare beams
    lattices = [Lattice(opt.max_token_seq_len, beam_size) for _ in range(batch_size)]

    #- Decode
    for i in range(opt.max_token_seq_len):
        len_dec_seq = i + 1

        if_finish = True
        lattice_index = []
        dec_partial_seq = []
        for i in range(batch_size):
            lattice = lattices[i]
            if not lattice.done:
                if_finish = False
                results, weights = lattice.get_results(mode='active')
                dec_partial_seq += results
                lattice_index += [i] * len(results)
        if if_finish:
            break
        #size: dec_partial_seq(num of results(sum of each beam) * seq_len)
        #       lattice_index(num of results(sum of each beam))

        dec_partial_seq = torch.LongTensor(dec_partial_seq)
        dec_partial_seq_mask = torch.ones(dec_partial_seq.size(), dtype=torch.long) #generate a mask for decoder, actually it's useless in decoding, and can be optimized
        lattice_index = torch.LongTensor(lattice_index)

        if opt.use_gpu:
            dec_partial_seq = dec_partial_seq.cuda()
            dec_partial_seq_mask = dec_partial_seq_mask.cuda()
            lattice_index = lattice_index.cuda()
        # prepare for parallelize decoding
        curr_src_pad_mask = src_pad_mask.index_select(0, lattice_index)
        curr_enc_output = enc_output.index_select(0, lattice_index)

        # -- Parallelize Decoding -- #
        dec_output, *_ = model.decoder(dec_partial_seq, dec_partial_seq_mask, curr_src_pad_mask, curr_enc_output)
        dec_output = dec_output[:, -1, :] # (batch * beam) * d_model
        word_lk = model.prob_projection(dec_output)

        # advande lattice
        end = 0
        word_lk = word_lk.cpu().detach().numpy()
        for lattice in lattices:
            if lattice.done:
                continue
            start = end
            end += lattice.num_curr_active
            curr_word_lk = word_lk[start:end]
            lattice.advance(curr_word_lk)

    final_sequences = []
    final_weights = []
    for lattice in lattices:
        results, weights = lattice.get_results(mode='all')
        final_sequences += [results[:opt.nbest]]
        final_weights += [weights]

    return final_sequences, final_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_data_dir', required=True)
    parser.add_argument('-read_vocab_file', required=True)
    parser.add_argument('-load_model_file', required=True)
    parser.add_argument('-save_result_file', required=True)

    parser.add_argument('-max_token_seq_len', type=int, required=True)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-beam_size', type=int, default=20)
    # if nbest > 2, it will produce result with repeat key (as optional result for rescoring)
    parser.add_argument('-nbest', type=int, default=10)    
    parser.add_argument('-use_gpu', action='store_true')
    opt = parser.parse_args()

    if opt.use_gpu:
        available_gpu_ids = get_available_gpu_ids()
        if len(available_gpu_ids) == 0:
            print('[ERROR] no cuda device available!')
            exit(1)
        else:
            torch.cuda.set_device(available_gpu_ids[0])
            print('[INFO] use gpu device {}'.format(available_gpu_ids[0]))

    if opt.nbest > opt.beam_size:
        print("[ERROR] nbest should not larger than beam_size")
        exit(1)

    checkpoint = torch.load(opt.load_model_file, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model_options = checkpoint['model_options']
    if opt.use_gpu:
        model = model.cuda()
    model.prob_projection = nn.LogSoftmax(dim=1)
    print('[INFO] loading model with parameter: {}'.format(model_options))


    decode_data = initialize_batch_loader(opt.read_data_dir + '/feats.scp', opt.read_data_dir + '/text', opt.read_vocab_file, opt.batch_size)
    print('[INFO] batch loader is initialized')


    word2idx = instances_handler.read_vocab(opt.read_vocab_file)
    idx2word = {index:word for word, index in word2idx.items()}
    with open(opt.save_result_file, 'w', encoding='utf-8') as f:
        for batch in tqdm(decode_data, mininterval=2, desc='(decode)'):
            all_hyp, all_scores = translate_batch(model, batch, opt, model_options)
            key = batch[0]
            for (k, t, scores) in zip(key, all_hyp, all_scores):
                t = [[idx2word[index] if index in idx2word else constants.UNK_WORD for index in i[1:-1]] for i in t]
                t = [' '.join(i) for i in t]
                for (line, score) in zip(t, scores):
                    f.write(k + '\t' + str(score) + '\t' + line + '\n')


if __name__ == '__main__':
    main()