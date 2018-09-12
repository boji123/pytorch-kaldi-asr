#!/bin/bash

export train_cmd="./queue.pl -q CPU_QUEUE -l ram_free=3G,mem_free=3G,io=3.125"
#export decode_cmd="run.pl -q CPU_QUEUE -l ram_free=3G,mem_free=3G,io=3.125"
export cuda_cmd="./queue.pl -q GPU_QUEUE@@amax2017 -l gpu=1"
export cuda_cmd="./queue.pl -q GPU_QUEUE@compute-0-5.local -l gpu=1"

set -e # exit on error

type=token
if [ $type == "token" ];then
	stage=1
	data=data/xunfei_token/haircare
        dir=exp/xunfei_token/haircare
        mkdir -p $dir
	if [ $stage -le 1 ];then
        	echo "[Stage] using jieba to cut the data..."
		python3 tools/jieba_cutter.py -sents_input data/xunfei_token/haircare_q_20180807.txt -cutted_output $data/src.txt
		#注意这里的数据处理，已经把制表符换成了空格了,与源数据不一样
		cp data/xunfei_token/haircare_token_20180807.txt $data/tgt.txt
	fi
	#exit 0	
	if [ $stage -le 2 ];then
		echo "[Stage] dividing dataset..."
		python3 tools/divide_train_valid.py -src $data/src.txt -tgt $data/tgt.txt -valid_rate 0.1 -output_dir $data
		echo "[Stage] processing data..."
       		python3 preprocess.py -min_word_count 0 -train_src $data/train_src.txt -train_tgt $data/train_tgt.txt -valid_src $data/valid_src.txt -valid_tgt $data/valid_tgt.txt -save_data $dir/data.pack
        fi
	
	if [ $stage -le 3 ];then
		echo "[Stage] training... log is in $dir/train.log"
		mkdir -p $dir/model
        	$cuda_cmd $dir/train.log CUDA_VISIBLE_DEVICES=3 python3 train.py \
			-data $dir/data.pack \
			-save_model $dir/model/trained \
			-save_mode best \
			-proj_share_weight \
			-batch_size 50 \
			-epoch 50 \
			\
			-n_layers 3 \
			-n_head 4 \
			-d_model 256 \
			-d_inner_hid 512 \
			-d_k 64 \
			-d_v 64 \
			-dropout 0.1
        fi

	if [ $stage -le 4 ];then
		echo "[Stage] testing... log is in $dir/translate.log"
        	$cuda_cmd $dir/translate.log CUDA_VISIBLE_DEVICES=3 python3 translate.py -model $dir/model/trained.chkpt -vocab $dir/data.pack -src $data/valid_src.txt -output $dir/pred.txt
	fi
fi
