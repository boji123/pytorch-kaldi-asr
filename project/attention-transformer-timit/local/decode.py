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
from transformer.Beam import Beam

from train import initialize_batch_loader

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

    #--- Repeat data for beam
    src_seq = src_seq.data.repeat(1, beam_size, 1).view(src_seq.size(0) * beam_size, src_seq.size(1), src_seq.size(2))
    src_pad_mask = src_pad_mask.data.repeat(1, beam_size).view(src_pad_mask.size(0) * beam_size, src_pad_mask.size(1))
    enc_output = enc_output.data.repeat(1, beam_size, 1).view(enc_output.size(0) * beam_size, enc_output.size(1), enc_output.size(2))

    #--- Prepare beams
    beams = [Beam(beam_size, opt.use_gpu) for _ in range(batch_size)]
    beam_inst_idx_map = {beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))}
    n_remaining_sents = batch_size

    #- Decode
    for i in range(opt.max_token_seq_len):
        len_dec_seq = i + 1
        # -- Preparing decoded data seq -- #
        # size: batch x beam x seq
        dec_partial_seq = torch.stack([b.get_current_state() for b in beams if not b.done])
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq).cpu().numpy()

        dec_partial_seq = torch.LongTensor(dec_partial_seq)
        dec_partial_seq_mask = torch.ones(dec_partial_seq.size(), dtype=torch.long) #generate a mask for decoder, actually it's useless in decoding, and can be optimized

        if opt.use_gpu:
            dec_partial_seq = dec_partial_seq.cuda()
            dec_partial_seq_mask = dec_partial_seq_mask.cuda()

        # -- Decoding -- #
        dec_output, *_ = model.decoder(dec_partial_seq, dec_partial_seq_mask, src_pad_mask, enc_output)
        dec_output = dec_output[:, -1, :] # (batch * beam) * d_model
        out = model.prob_projection(dec_output)

        # batch x beam x n_words
        word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

        active_beam_idx_list = []
        for beam_idx in range(batch_size):
            if beams[beam_idx].done:
                continue

            inst_idx = beam_inst_idx_map[beam_idx]
            if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                active_beam_idx_list += [beam_idx]

        if not active_beam_idx_list:
            # all instances have finished their path to <EOS>
            break

        # in this section, the sentences that are still active are
        # compacted so that the decoder is not run on completed sentences
        active_inst_idxs = torch.LongTensor([beam_inst_idx_map[k] for k in active_beam_idx_list])
        if opt.use_gpu:
            active_inst_idxs = active_inst_idxs.cuda()
        # update the idx mapping
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

        def update_active_seq(seq_var, active_inst_idxs):
            ''' Remove the src sequence of finished instances in one batch. '''

            inst_idx_dim_size, *rest_dim_sizes = seq_var.size()
            inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
            new_size = (inst_idx_dim_size, *rest_dim_sizes)

            # select the active instances in batch
            original_seq_data = seq_var.data.view(n_remaining_sents, -1)
            active_seq_data = original_seq_data.index_select(0, active_inst_idxs)
            active_seq_data = active_seq_data.view(*new_size)

            return active_seq_data

        def update_active_enc_info(enc_info_var, active_inst_idxs):
            ''' Remove the encoder outputs of finished instances in one batch. '''

            inst_idx_dim_size, *rest_dim_sizes = enc_info_var.size()
            inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
            new_size = (inst_idx_dim_size, *rest_dim_sizes)

            # select the active instances in batch
            original_enc_info_data = enc_info_var.data.view(
                n_remaining_sents, -1, model_options.d_model)
            active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
            active_enc_info_data = active_enc_info_data.view(*new_size)

            return active_enc_info_data

        src_pad_mask = update_active_seq(src_pad_mask, active_inst_idxs)
        enc_output = update_active_enc_info(enc_output, active_inst_idxs)

        #- update the remaining size
        n_remaining_sents = len(active_inst_idxs)


    #- Return useful information
    all_hyp, all_scores = [], []
    for beam_idx in range(batch_size):
        scores, tail_idxs = beams[beam_idx].sort_scores()
        all_scores += [[k.item() for k in scores[:opt.nbest]]]
        hyps = [[k.item() for k in beams[beam_idx].get_hypothesis(i)] for i in tail_idxs[:opt.nbest]]
        all_hyp += [hyps]

    return all_hyp, all_scores


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
            tgt_seq = batch[3]
            for (k, s, t, scores) in zip(key, tgt_seq, all_hyp, all_scores):
                s = list(s[np.where(s>3)]) #remove function label 0,1,2,3 (currently, 0 as blank, 2 as start, 3 as end)

                s = [idx2word[index] if index in idx2word else constants.UNK_WORD for index in s]
                t = [[idx2word[index] if index in idx2word else constants.UNK_WORD for index in i] for i in t]

                s = ' '.join(s)
                t = [' '.join(i) for i in t]
                for (line, score) in zip(t, scores):
                    f.write(k + '\t' + str(score) + '\t' + line + '\n')


if __name__ == '__main__':
    main()