#!/bin/bash
#------------------------------------------------------------
#edited by baiji
#an example for pytorch training
#------------------------------------------------------------
#--------------------    trainning cmd   --------------------
export train_cmd="./queue.pl -q CPU_QUEUE -l ram_free=3G,mem_free=3G,io=3.125"
export cuda_cmd="./queue.pl -q GPU_QUEUE@@amax2017 -l gpu=1"
export cuda_cmd="./queue.pl -q GPU_QUEUE@compute-0-5.local -l gpu=1"
. ./path.sh
set -e # exit on error
#------------------------------------------------------------


#notice: step of data preparation here is done by kaldi, so I just copy the data files to data/
if [ -d data ]&&[ -f data/feats.maxlen_500.scp ]&&[ -f data/text.maxlen_500 ]; then
    echo '[INFO] trainning data founded, continue trainning process.'
else
    echo '[ERROR] requir file not founded, please check your directory'
    exit 1
fi

stage=0
if [ $stage -le 0 ]; then
    python3 local/prepare_vocab.py -read_instances_file data/text.maxlen_500 -save_vocab_file exp/vocab.torch
fi

if [ $stage -le 1 ]; then
    #read_feats_scp_file and read_vocab_file for initializing the input and output dimension
    python3 local/initialize_model.py \
        -read_feats_scp_file data/feats.maxlen_500.scp \
        -read_vocab_file exp/vocab.torch \
        -max_token_seq_len 50 \
        \
        -n_layers 4 \
        -n_head 6 \
        -d_model 512 \
        -d_inner_hid 512 \
        -d_k 64 \
        -d_v 64 \
        -dropout 0.1 \
        \
        -save_model_file exp/model.init.torch
fi

if [ $stage -le 2 ]; then
    $cuda_cmd train.log CUDA_VISIBLE_DEVICES=3 python3 -u local/train.py \
        -read_feats_scp_file data/feats.maxlen_500.scp \
        -read_text_file data/text.maxlen_500 \
        -read_vocab_file exp/vocab.torch \
        -load_model_file exp/model.init.torch \
        -batch_size 16 \
        -save_model_perfix exp/model \
        -use_gpu
fi
