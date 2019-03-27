''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from utils import constants
from transformer.Modules import BottleLinear as Linear
from transformer.Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F

from TDNN import TDNNLayer
from TDNN import ConcatLayer
from TDNN import LDALayer
# further edited by liu.baiji
# adapted for speech recognition

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk

    return pad_attn_mask

def get_attn_subsequent_mask(seq, start, end):
    '''function for generating time restrict mask.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))

    mask_start = 1 - np.triu(np.ones(attn_shape), k=start).astype('uint8')
    mask_end = np.triu(np.ones(attn_shape), k=end + 1).astype('uint8')
    mask = mask_start + mask_end
    mask = torch.from_numpy(mask)
    if seq.is_cuda:
        mask = mask.cuda()
    return mask

def fold_seq_and_mask(seq, pad_mask, fold):
    if fold == 1:
        return seq, pad_mask
    elif fold < 1:
        print('[ERROR] invaldi data fold parameter')
        exit(1)
    else:
        #reshape the input: length/2, dim*2
        seq_len_trimed = seq.size(1) - seq.size(1) % fold
        seq = seq[:,:seq_len_trimed].contiguous()
        seq = seq.view(seq.size(0),-1,seq.size(2)*fold)

        #resample the mask as size of input
        pad_mask = pad_mask[:,fold-1::fold].contiguous()
        return seq, pad_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(
            self, n_src_dim, encoder_max_len, n_layers=2, n_head=3, sub_sequence=(-1,1),
            d_k=64, d_v=64, d_model=256, d_inner_hid=256, dropout=0.1):

        super(Encoder, self).__init__()
        self.sub = sub_sequence

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.position_enc = nn.Embedding(encoder_max_len, d_model, padding_idx=constants.PAD)
        self.position_enc.weight.data = position_encoding_init(encoder_max_len, d_model)
        self.position_enc.weight.requires_grad = False

        self.trans_pos_enc = nn.Embedding(encoder_max_len, d_model, padding_idx=constants.PAD)
        self.trans_pos_enc.weight.data = position_encoding_init(encoder_max_len, d_model)
        self.trans_pos_enc.weight.requires_grad = False

        #project the source to dim of model
        self.src_projection = Linear(n_src_dim, d_model, bias=False)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pad_mask, return_attns=False):
        src_pos = torch.arange(0, src_seq.size(1)).long().repeat(src_seq.size(0), 1)
        if src_seq.is_cuda:
            src_pos = src_pos.cuda()
        trans_pos = self.trans_pos_enc(src_pos)
        src_pos = self.position_enc(src_pos)

        #src_seq batch*len*featdim -> batch*len*modeldim
        src_seq = self.src_projection(src_seq)

        if return_attns:
            enc_slf_attns = []
        
        enc_output = src_seq + src_pos
        enc_output = self.dropout(enc_output)

        enc_slf_attn_pad_mask = get_attn_padding_mask(src_pad_mask, src_pad_mask)
        enc_slf_attn_sub_mask = get_attn_subsequent_mask(src_pad_mask, self.sub[0], self.sub[1])
        enc_slf_attn_mask = torch.gt(enc_slf_attn_pad_mask + enc_slf_attn_sub_mask, 0)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        enc_output = enc_output + trans_pos
        enc_output = self.dropout(enc_output)
        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output,


class EncoderTest(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, lda_mat, n_src_dim, encoder_max_len, d_model=256, dropout=0.1, contexts=[[0]]):

        super(EncoderTest, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        self.trans_pos_enc = nn.Embedding(encoder_max_len, d_model, padding_idx=constants.PAD)
        self.trans_pos_enc.weight.data = position_encoding_init(encoder_max_len, d_model)
        self.trans_pos_enc.weight.requires_grad = False

        #project the source to dim of model
        lda_concat_index = [-2,-1,0,1,2]
        self.concat = ConcatLayer(lda_concat_index)
        self.lda_layer = LDALayer(lda_mat)
        self.src_projection = Linear(n_src_dim * len(lda_concat_index), d_model, bias=False)
        #self.tdnn0 = TDNNLayer(n_src_dim * len(lda_concat_index), d_model, contexts[0], dropout=dropout)
        self.tdnn_stack = nn.ModuleList([TDNNLayer(d_model, d_model, context, dropout=dropout) for context in contexts])

    def forward(self, src_seq, src_pad_mask):
        src_pos = torch.arange(0, src_seq.size(1)).long().repeat(src_seq.size(0), 1)
        if src_seq.is_cuda:
            src_pos = src_pos.cuda()
        trans_pos = self.trans_pos_enc(src_pos)

        #applying lda
        src_seq = self.lda_layer(self.concat(src_seq))
        #src_seq batch*len*featdim -> batch*len*modeldim
        src_seq = self.src_projection(src_seq)
        enc_output = self.dropout(src_seq)

        #TDNN
        #src_seq = self.tdnn0(src_seq)
        for tdnn_layer in self.tdnn_stack:
            enc_output = tdnn_layer(enc_output)

        enc_output = enc_output + trans_pos
        enc_output = self.dropout(enc_output)
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, decoder_max_len, n_layers=2, n_head=3, sub_sequence=(-1,1),
            d_k=64, d_v=64, en_d_model=256, de_d_model=128, d_inner_hid=128, dropout=0.1):

        super(Decoder, self).__init__()
        self.sub = sub_sequence

        self.en_d_model = en_d_model
        self.de_d_model = de_d_model
        self.dropout = nn.Dropout(dropout)
        self.position_enc = nn.Embedding(decoder_max_len, de_d_model, padding_idx=constants.PAD)
        self.position_enc.weight.data = position_encoding_init(decoder_max_len, de_d_model)
        self.position_enc.weight.requires_grad = False

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, de_d_model, padding_idx=constants.PAD)
        self.tgt_word_proj = Linear(de_d_model, n_tgt_vocab, bias=False)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(de_d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        #project the encoder output to dim of decoder
        self.enc_dec_projection = Linear(en_d_model, de_d_model, bias=False)

    def forward(self, tgt_seq, tgt_pad_mask, src_pad_mask, enc_output, return_attns=False):
        tgt_pos = torch.arange(0, tgt_seq.size(1)).long().repeat(tgt_seq.size(0), 1)
        if tgt_seq.is_cuda:
            tgt_pos = tgt_pos.cuda()
        tgt_pos = self.position_enc(tgt_pos)
        enc_output = self.enc_dec_projection(enc_output)
        #word -> dim model
        # warning: embedding require long tensor, maybe it's a waste of menory, waiting to be solved
        tgt_seq = self.tgt_word_emb(tgt_seq)

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_pad_mask, tgt_pad_mask)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_pad_mask, self.sub[0], self.sub[1])
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_pad_mask, src_pad_mask)
        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []

        dec_output = tgt_seq + tgt_pos
        
        #dec_output = tgt_seq
        dec_output = self.dropout(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]
        dec_output = self.dropout(dec_output)
        dec_output = self.tgt_word_proj(dec_output)
        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, n_src_dim, n_tgt_vocab, lda_mat, encoder_max_len, decoder_max_len, src_fold=1, encoder_sub_sequence=(-100,0), decoder_sub_sequence=(-20,0),
            en_layers=2, de_layers=2, n_head=3, en_d_model=256, de_d_model=128, d_k=64, d_v=64, en_dropout=0.2, de_dropout=0.2, tdnn_contexts=[[0]]):

        super(Transformer, self).__init__()

        self.src_fold = src_fold
        '''
        self.encoder = Encoder(
            n_src_dim=n_src_dim * self.src_fold, encoder_max_len=encoder_max_len, sub_sequence=(-100,0),
            n_layers=en_layers, n_head=n_head, d_model=en_d_model, d_inner_hid=en_d_model, dropout=dropout)
        '''

        self.encoder_test = EncoderTest(lda_mat=lda_mat, n_src_dim=n_src_dim * self.src_fold, encoder_max_len=encoder_max_len, d_model=en_d_model, dropout=en_dropout, contexts=tdnn_contexts)

        self.decoder = Decoder(n_tgt_vocab=n_tgt_vocab, decoder_max_len=decoder_max_len, sub_sequence=decoder_sub_sequence,
            n_layers=de_layers, n_head=n_head, en_d_model=en_d_model, de_d_model=de_d_model ,d_inner_hid=de_d_model, dropout=de_dropout)

    def forward(self, src_seq, src_pad_mask, tgt_seq, tgt_pad_mask):
        #reshape the input and input mask: length/fold, dim*fold
        src_seq, src_pad_mask = fold_seq_and_mask(src_seq, src_pad_mask, self.src_fold)

        #enc_output, *_ = self.encoder_test(src_seq, src_pad_mask)
        enc_output = self.encoder_test(src_seq, src_pad_mask)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pad_mask, src_pad_mask, enc_output)
        #return batch*seq len*word dim
        return dec_output