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
export cuda_cmd="queue.pl -q GPU_QUEUE@compute-0-5.local -l gpu=1,io=0"
set -e # exit on error
#------------------------------------------------------------
stage=4
if [ $stage -le 0 ]; then
    echo '[PROCEDURE] preparing instances.'
    max_len=500
    for dataset in train dev test; do
        #feat-to-len is a kaldi src file, you need to export the path
        feat-to-len scp:data/$dataset/feats.scp ark,t:data/$dataset/feats.length
        #require feats.scp feats.length text
        python3 local/trim_instance_length.py -data_dir data/$dataset -output_dir data/${dataset}_filtered -max_len $max_len
    done
fi


if [ $stage -le 1 ]; then
    echo '[PROCEDURE] preparing vocabulary for output label'
    mkdir -p exp
    python3 local/prepare_vocab.py -read_instances_file data/train/text -save_vocab_file exp/vocab.torch
fi


if [ $stage -le 2 ]; then
    echo '[PROCEDURE] reading dimension from data file and initialize the model'
    #read_feats_scp_file and read_vocab_file for initializing the input and output dimension
    python3 local/initialize_model.py \
        -read_feats_scp_file data/train/feats.scp \
        -read_vocab_file exp/vocab.torch \
        -save_model_file exp/model.init.torch \
        \
        -n_layers 2 \
        -n_head 3 \
        -d_model 256 \
        -d_inner_hid 256 \
        -d_k 64 \
        -d_v 64 \
        -dropout 0.1 \

fi


use_gpu=true
if [ $stage -le 3 ]; then
    echo '[PROCEDURE] trainning start... log is in train.log'
    time=$(date "+%Y%m%d-%H%M%S")
    mkdir -p exp/model-$time
    if $use_gpu; then
        $cuda_cmd train.log CUDA_VISIBLE_DEVICES=2 python3 -u local/train.py \
            -read_train_dir data/train_filtered \
            -read_dev_dir data/dev_filtered \
            -read_test_dir data/test_filtered \
            -read_vocab_file exp/vocab.torch \
            -load_model_file exp/model.init.torch \
            \
            -optim_start_lr 0.001 \
            -optim_soft_coefficient 1000 \
            -epoch 50 \
            -batch_size 50 \
            -save_model_dir exp/model-$time \
            -use_gpu || exit 1
    else
        python3 -u local/train.py \
            -read_train_dir data/train_filtered \
            -read_dev_dir data/dev_filtered \
            -read_test_dir data/test_filtered \
            -read_vocab_file exp/vocab.torch \
            -load_model_file exp/model.init.torch \
            \
            -optim_start_lr 0.001 \
            -optim_soft_coefficient 1000 \
            -epoch 50 \
            -batch_size 50 \
            -save_model_dir exp/model-$time || exit 1
    fi
    echo '[INFO]trainning finish.'
fi


if [ $stage -le 4 ]; then
    echo '[PROCEDURE] decoding test set... log is in decode.log'
    $cuda_cmd decode.log CUDA_VISIBLE_DEVICES=3 python3 -u local/decode.py \
        -read_decode_dir data/train_filtered \
        -read_vocab_file exp/vocab.torch \
        -load_model_file exp/model.train96 \
        -max_token_seq_len 100 \
        -batch_size 32 \
        -beam_size 10 \
        -save_result_file exp/result.txt\
        -use_gpu || exit 1
    echo '[INFO] decoding finish.'
fi
