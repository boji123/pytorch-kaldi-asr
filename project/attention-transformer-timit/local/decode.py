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
from transformer.Beam import Beam

def initialize_batch_loader(read_feats_scp_file, read_text_file, read_vocab_file, batch_size):
    utterances = {}
    with open(read_feats_scp_file, encoding='utf-8') as file:
        for line in file:
            (key,rxfile) = line.split()
            utterances[key] = rxfile
            #the utterance can be read by 
            #mat = kaldi_io.read_mat(rxfile)

    print('[INFO] get {} utterances from {}.'.format(len(utterances), read_feats_scp_file))

    label_text = {}
    with open(read_text_file, encoding='utf-8') as file:
        for line in file:
            data = line.split()
            key = data[0]
            text = data[1:]
            label_text[key] = text
    print('[INFO] get {} labels from {}.'.format(len(label_text), read_text_file))

    #begin & end of sequence
    label_text = instances_handler.add_control_words(label_text)
    label = instances_handler.apply_vocab(label_text, read_vocab_file, 'word2idx')

    #connect the label and utterances
    trainning_triples = []
    for key in utterances:
        if key in label:
            triples = (key, utterances[key], label[key])
            trainning_triples.append(triples)
    print('[INFO] match {} utterance-label pairs.'.format(len(trainning_triples)))

    #in batch loader, trainning feature will be loaded while itering
    #it increase io but decrease memory use
    batch_loader = BatchLoader(trainning_triples, batch_size, print_info = False)
    return batch_loader


def translate_batch(model, batch, opt, model_options):
    model.eval()
    # prepare data
    #key = [triples[0] for triples in batch]
    src = [triples[1] for triples in batch]
    tgt = [triples[2] for triples in batch]
    src_seq, src_pad_mask = instances_handler.pad_to_longest(src)
    tgt_seq, tgt_pad_mask = instances_handler.pad_to_longest(tgt)

    src_seq = Variable(torch.FloatTensor(src_seq)) #batch * max length in batch * padded feature dim
    src_pad_mask = Variable(torch.LongTensor(src_pad_mask)) #batch * maxlength in batch * bool mask dim
    tgt_seq = Variable(torch.LongTensor(tgt_seq)) #batch * max length in batch * padded index dim
    tgt_pad_mask = Variable(torch.LongTensor(tgt_pad_mask)) #batch * maxlength in batch * bool mask dim

    if opt.use_gpu:
        src_seq = src_seq.cuda()
        src_pad_mask = src_pad_mask.cuda()
        tgt_seq = tgt_seq.cuda()
        tgt_pad_mask = tgt_pad_mask.cuda()

    goal = tgt_seq[:, 1:]
    tgt_seq = tgt_seq[:, :-1]
    tgt_pad_mask = tgt_pad_mask[:, :-1]

    beam_size = opt.beam_size
    batch_size = src_seq.size(0)
    #---------------------------------------------------------------------------------------
    #- Enocde
    enc_output, *_ = model.encoder(src_seq, src_pad_mask)

    #--- Repeat data for beam
    src_seq = Variable(
        src_seq.data.repeat(1, beam_size, 1).view(
            src_seq.size(0) * beam_size, src_seq.size(1), src_seq.size(2)))

    src_pad_mask = Variable(
        src_pad_mask.data.repeat(1, beam_size).view(
            src_pad_mask.size(0) * beam_size, src_pad_mask.size(1)))

    enc_output = Variable(
        enc_output.data.repeat(1, beam_size, 1).view(
            enc_output.size(0) * beam_size, enc_output.size(1), enc_output.size(2)))

    #--- Prepare beams
    beams = [Beam(beam_size, opt.use_gpu) for _ in range(batch_size)]
    beam_inst_idx_map = {
        beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))}
    n_remaining_sents = batch_size

    #- Decode
    for i in range(opt.max_token_seq_len):
        len_dec_seq = i + 1
        # -- Preparing decoded data seq -- #
        # size: batch x beam x seq
        dec_partial_seq = torch.stack([b.get_current_state() for b in beams if not b.done])
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq).cpu().numpy()
        dec_partial_seq, dec_partial_seq_mask = instances_handler.pad_to_longest(dec_partial_seq)

        # size: (batch * beam) x seq
        dec_partial_seq = Variable(torch.LongTensor(dec_partial_seq))
        dec_partial_seq_mask = Variable(torch.LongTensor(dec_partial_seq_mask))

        if opt.use_gpu:
            dec_partial_seq = dec_partial_seq.cuda()
            dec_partial_seq_mask = dec_partial_seq_mask.cuda()
            
        print('\nnext\n')
        print(dec_partial_seq.size())
        print(dec_partial_seq_mask.size())
        print(src_pad_mask.size())
        print(enc_output.size())
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

            return Variable(active_seq_data, volatile=True)

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

            return Variable(active_enc_info_data, volatile=True)

        src_seq = update_active_seq(src_seq, active_inst_idxs)
        enc_output = update_active_enc_info(enc_output, active_inst_idxs)

        #- update the remaining size
        n_remaining_sents = len(active_inst_idxs)

    #- Return useful information
    all_hyp, all_scores = [], []

    for beam_idx in range(batch_size):
        scores, tail_idxs = beams[beam_idx].sort_scores()
        all_scores += [scores[0]]

        hyps = [beams[beam_idx].get_hypothesis(i) for i in tail_idxs[0]]
        all_hyp += [hyps]

    return all_hyp, all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-read_decode_dir', required=True)
    parser.add_argument('-read_vocab_file', required=True)
    parser.add_argument('-load_model_file', required=True)
    parser.add_argument('-save_result_file', required=True)

    parser.add_argument('-max_token_seq_len', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-beam_size', type=int, default=10)
    parser.add_argument('-use_gpu', action='store_true')
    opt = parser.parse_args()


    test_data = initialize_batch_loader(opt.read_decode_dir + '/feats.scp', opt.read_decode_dir + '/text', opt.read_vocab_file, opt.batch_size)
    print('[INFO] batch loader is initialized')

    checkpoint = torch.load(opt.load_model_file)
    model = checkpoint['model']
    model_options = checkpoint['model_options']
    print('[INFO] loading model with parameter: {}'.format(model_options))

    model.prob_projection = nn.LogSoftmax()

    if opt.use_gpu:
        model = model.cuda()

    with open(opt.save_result_file, 'wb') as f:
        for batch in tqdm(test_data, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translate_batch(model, batch, opt, model_options)
            exit(0)
            '''
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ' '.join([test_data.tgt_idx2word[idx] for idx in idx_seq]) + '\n'
                    pred_line = pred_line.encode('utf-8')
                    f.write(pred_line)
            '''

if __name__ == '__main__':
    main()