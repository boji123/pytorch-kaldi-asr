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
export cuda_cmd="queue.pl -q GPU_QUEUE@compute-0-6.local -l gpu=1,io=0,ram_free=1G"
set -e # exit on error
#------------------------------------------------------------
use_gpu=true
cuda_device=0,1,2,3
stage=6
model_suffix=_enL3d25b
#------------------------------------------------------------
#data_perfix=
data_perfix=_hires
#speed_perturb=
speed_perturb=_sp
lang=data/language

if [ $stage -le 0 ]; then
    echo '[PROCEDURE] preparing instances.'
    max_len=500
    for dataset in train${speed_perturb}${data_perfix} dev${speed_perturb}${data_perfix} test${speed_perturb}${data_perfix}; do
        #feat-to-len is a kaldi src file, you need to export the path
        feat-to-len scp:data/$dataset/feats.scp ark,t:data/$dataset/feats.length
        #require feats.scp feats.length text
        PYTHONIOENCODING=utf-8 python3 local/trim_instance_length.py -data_dir data/$dataset -output_dir data/${dataset}_filtered -max_len $max_len
    done
fi

if [ $stage -le 1 ]; then
    echo '[PROCEDURE] preparing vocabulary for output label'
    mkdir -p ${lang}
    python3 local/prepare_vocab.py -read_instances_file data/train${speed_perturb}${data_perfix}/text -save_vocab_file ${lang}/vocab.txt
    #add the disambig symbol, for generating fst file
    index=`wc -l test.txt | cut -d' ' -f1`
    echo "#0 ${index}" >> vocab.txt
fi

if [ $stage -le 2 ]; then
    echo '[PROCEDURE] preparing language model(arpa) and fst file.'
    cat data/train${data_perfix}/text | cut -d' ' -f2- |\
        ngram-count -text - -order 3 -lm ${lang}/lm.3k.gz
    #in this project, <blank> as eps symbol, #0 as disambig symbol
    gunzip -c ${lang}/lm.3k.gz | arpa2fst --disambig-symbol=#0 --read-symbol-table=${lang}/vocab.txt - ${lang}/lm.3k.fst
fi
#exit 0
#------------------------------------------------------------
time=$(date "+%Y%m%d-%H%M%S")
model_dir=exp/model${model_suffix}-${time}
if [ $stage -le 3 ]; then
    echo '[PROCEDURE] reading dimension from data file and initialize the model'
    mkdir -p $model_dir
    #read_feats_scp_file and read_vocab_file for initializing the input and output dimension
    PYTHONIOENCODING=utf-8 python3 local/initialize_model.py \
        -read_feats_scp_file data/train${speed_perturb}${data_perfix}_filtered/feats.scp \
        -read_vocab_file ${lang}/vocab.txt \
        -save_model_file ${model_dir}/model.init \
        \
        -encoder_max_len 500 \
        -decoder_max_len 80 \
        -src_fold 2 \
        -encoder_sub_sequence '(-100,0)' \
        -decoder_sub_sequence '(-20,0)' \
        \
        -en_layers 3 \
        -de_layers 2 \
        -n_head 3 \
        -d_model 256 \
        -d_inner_hid 256 \
        -d_k 64 \
        -d_v 64 \
        -dropout 0.25
fi

if [ $stage -le 4 ]; then
    echo '[PROCEDURE] trainning start... log is in train.log'
    if $use_gpu; then
        #attention: for keeping it same as origin one, the dev and test set should'n apply speed perturb
        $cuda_cmd train${model_suffix}.log CUDA_VISIBLE_DEVICES=${cuda_device} PYTHONIOENCODING=utf-8 python3 -u local/train.py \
            -read_train_dir data/train${speed_perturb}${data_perfix}_filtered \
            -read_dev_dir data/dev${data_perfix}_filtered \
            -read_test_dir data/test${data_perfix}_filtered \
            -read_vocab_file ${lang}/vocab.txt \
            -load_model_file ${model_dir}/model.init \
            \
            -optim_start_lr 0.001 \
            -optim_soft_coefficient 20000 \
            -epoch 300 \
            -batch_size 90 \
            -save_model_dir $model_dir \
            -save_interval 10 \
            -use_gpu || exit 1
    else
        PYTHONIOENCODING=utf-8 python3 -u local/train.py \
            -read_train_dir data/train${speed_perturb}${data_perfix}_filtered \
            -read_dev_dir data/dev${data_perfix}_filtered \
            -read_test_dir data/test${data_perfix}_filtered \
            -read_vocab_file ${lang}/vocab.txt \
            -load_model_file ${model_dir}/model.init \
            \
            -optim_start_lr 0.001 \
            -optim_soft_coefficient 5000 \
            -epoch 1 \
            -batch_size 90 \
            -save_model_dir $model_dir || exit 1
    fi
    echo '[INFO] trainning finish.'
fi

if [ $stage -le 5 ]; then
    echo '[PROCEDURE] combining model... log is in combine.log'
    num_combine=10 # num_combine = num_interval here
    #model_dir=
    model_list=`ls ${model_dir} --sort=time | grep ^epoch.*.torch$ | head -${num_combine}`

    if $use_gpu; then
        $cuda_cmd combine${model_suffix}.log CUDA_VISIBLE_DEVICES=${cuda_device} PYTHONIOENCODING=utf-8 python3 -u local/combine.py \
            -read_test_dir data/dev_hires_filtered \
            -read_vocab_file ${lang}/vocab.txt \
            -load_model_dir $model_dir \
            -load_model_file_list ${model_list} \
            -save_model_dir $model_dir \
            -use_gpu || exit 1
    else
        PYTHONIOENCODING=utf-8 python3 -u local/combine.py \
            -read_test_dir data/dev_hires_filtered \
            -read_vocab_file ${lang}/vocab.txt \
            -load_model_dir $model_dir \
            -load_model_file_list ${model_list} \
            -save_model_dir $model_dir || exit 1
    fi
    echo '[INFO] combining finish.'
fi

#------------------------------------------------------------
#decode & rescore
#------------------------------------------------------------
decode_dir=exp/decode_dev
data_dir=data/dev${data_perfix}_filtered
model_file=exp/model_enL3d25b-20190222-102224/best.epoch211.accu78.02.torch
if [ $stage -le 6 ]; then
    echo '[PROCEDURE] decoding test set... log is in decode.log'
    mkdir -p ${decode_dir}

    if $use_gpu; then
        $cuda_cmd decode.log CUDA_VISIBLE_DEVICES=${cuda_device} PYTHONIOENCODING=utf-8 python3 -u local/decode.py \
            -read_data_dir ${data_dir} \
            -read_vocab_file ${lang}/vocab.txt \
            -load_model_file ${model_file} \
            -max_token_seq_len 80 \
            -batch_size 32 \
            -beam_size 20 \
            -nbest 10\
            -save_result_file ${decode_dir}/decode.txt\
            -use_gpu || exit 1
    else
        PYTHONIOENCODING=utf-8 python3 -u local/decode.py \
            -read_data_dir ${data_dir} \
            -read_vocab_file ${lang}/vocab.txt \
            -load_model_file ${model_file} \
            -max_token_seq_len 80 \
            -batch_size 4 \
            -beam_size 10 \
            -nbest 1\
            -save_result_file ${decode_dir}/decode.txt || exit 1
    fi
    echo '[INFO] decoding finish.'
fi

if [ $stage -le 7 ]; then
    echo '[PROCEDURE] caculating language model score...'
    ngram -lm ${lang}/lm.3k.gz -order 3 -ppl ${decode_dir}/decode.txt -debug 1 | grep logprob | cut -d' ' -f4 > ${decode_dir}/lm.3k.score.txt
    sed -i '$d' ${decode_dir}/lm.3k.score.txt
    echo '[INFO] language model score computed.'

    mkdir -p ${decode_dir}/scoring
    PYTHONIOENCODING=utf-8 python3 -u local/rescore.py \
        -decode_file ${decode_dir}/decode.txt \
        -lm_score ${decode_dir}/lm.3k.score.txt \
        -inv_weight_list 10,11,12,13,13.5,14,14.5,15,15.5,16,16.5,17,18,19,20,1000 \
        -save_dir ${decode_dir}/scoring
    echo '[PROCEDURE] computing WER...'
    for rescore_file in `ls ${decode_dir}/scoring | grep rescore | grep -v wer`; do
        compute-wer --mode=present ark:${data_dir}/text ark:${decode_dir}/scoring/${rescore_file} \
            > ${decode_dir}/scoring/${rescore_file}_wer
    done

    echo '[INFO] best wer presented in file:'
    grep WER ${decode_dir}/scoring/*wer | best_wer.sh
fi
