#!/bin/bash
#------------------------------------------------------------
#edited by baiji
#an example for pytorch training
#------------------------------------------------------------
#--------------------    trainning cmd   --------------------
. ./path.sh
#queue.pl is in pytorch-kaldi-asr/kaldi
#it is edited to adapt the project path around line 373
export train_cmd="queue.pl -q CPU_QUEUE -l ram_free=3G,mem_free=3G,io=3.125"
export cuda_cmd="queue.pl -q GPU_QUEUE@@amax2017 -l gpu=1"
export cuda_cmd="queue.pl -q GPU_QUEUE@compute-0-5.local -l gpu=1"
set -e # exit on error
#------------------------------------------------------------
stage=0
if [ $stage -le 0 ]; then
    echo '[PROCEDURE] preparing instances.'
    max_len=500
    for dataset in train test; do
        #feat-to-len is a kaldi src file, you need to exporting the path
        feat-to-len scp:data/$dataset/feats.scp ark,t:data/$dataset/feats.length
        #require feats.scp feats.length text, output feats.filtered.scp and text.filtered
        python3 local/filter_instance_length.py -data_dir data/$dataset -max_len $max_len
    done
fi

if [ $stage -le 1 ]; then
    echo '[PROCEDURE] preparing vocabulary for output label'
    python3 local/prepare_vocab.py -read_instances_file data/text.filtered -save_vocab_file exp/vocab.torch
fi

if [ $stage -le 2 ]; then
    echo '[PROCEDURE] reading dimension from data file and initialize the model'
    #read_feats_scp_file and read_vocab_file for initializing the input and output dimension
    python3 local/initialize_model.py \
        -read_feats_scp_file data/feats.filtered.scp \
        -read_vocab_file exp/vocab.torch \
        -max_token_seq_len 50 \
        \
        -n_layers 3 \
        -n_head 4 \
        -d_model 256 \
        -d_inner_hid 256 \
        -d_k 64 \
        -d_v 64 \
        -dropout 0.1 \
        \
        -save_model_file exp/model.init.torch
fi

if [ $stage -le 3 ]; then
    echo '[PROCEDURE]trainning start... log is in train.log'
    $cuda_cmd train.log CUDA_VISIBLE_DEVICES=3 python3 -u local/train.py \
        -read_feats_scp_file data/feats.filtered.scp \
        -read_text_file data/text.filtered \
        -read_vocab_file exp/vocab.torch \
        -load_model_file exp/model.init.torch \
        -epoch 50 \
        -batch_size 32 \
        -save_model_perfix exp/model \
        -use_gpu
    echo '[INFO]trainning finish.'
fi
