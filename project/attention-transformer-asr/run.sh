#!/bin/bash
#----------------------------------------
#edited by baiji
#an example for pytorch training
#----------------------------------------
#----------        path        ----------
MY_PYTORCH_KALDI_DIR=$HOME/work/pytorch-kaldi-asr
export PATH=$MY_PYTORCH_KALDI_DIR:$MY_PYTORCH_KALDI_DIR/kaldi:$MY_PYTORCH_KALDI_DIR/pytorch:$PATH
export PYTHONPATH=$MY_PYTORCH_KALDI_DIR/kaldi:$MY_PYTORCH_KALDI_DIR/pytorch:$PYTHONPATH
#if a user want to change a file in the library, just copy it to the local directory.
#the program will search local first
USER_EDITED_LIBRARY=`pwd`/local
export PATH=$USER_EDITED_LIBRARY:$USER_EDITED_LIBRARY/kaldi:$USER_EDITED_LIBRARY/pytorch:$PATH
export PYTHONPATH=$USER_EDITED_LIBRARY/kaldi:$USER_EDITED_LIBRARY/pytorch:$PYTHONPATH
#----------    trainning cmd   ----------
export train_cmd="queue.pl -q CPU_QUEUE -l ram_free=3G,mem_free=3G,io=3.125"
export cuda_cmd="queue.pl -q GPU_QUEUE@@amax2017 -l gpu=1"
export cuda_cmd="queue.pl -q GPU_QUEUE@compute-0-5.local -l gpu=1"
#----------------------------------------
set -e # exit on error
#----------------------------------------


#notice: step of data preparation here is done by kaldi, so I just copy the data files to data/
if [ -d data ]&&[ -f data/feats.scp ]&&[ -f data/text ]; then
    echo '[INFO] trainning data founded, continue trainning process.'
else
    echo '[ERROR] requir file not founded, please check your directory'
    exit 1
fi

stage=99
if [ $stage -le 0 ]; then
    python3 local/prepare_vocab.py -read_instances_file data/text -save_vocab_file exp/vocab.torch
fi

if [ $stage -le 1 ]; then
    python3 local/initialize_model.py \
        -read_feats_scp_file data/feats.scp \
        -read_vocab_file exp/vocab.torch \
        -max_token_seq_len 50 \
        \
        -n_layers 6 \
        -n_head 8 \
        -d_word_vec 512 \
        -d_model 512 \
        -d_inner_hid 1024 \
        -d_k 64 \
        -d_v 64 \
        -dropout 0.1 \
        \
        -save_model_file exp/model.init
fi

if [ $stage -le 2 ]; then
    python3 local/train.py
fi

if [ $stage -eq 99 ]; then 
    python3 local/train.py \
        -read_feats_scp_file data/feats.scp \
        -read_text_file data/text \
        -read_vocab_file exp/vocab.torch
        
    exit 0
fi