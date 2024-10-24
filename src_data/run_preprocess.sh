#!/bin/bash
DOMAIN=$1
CACHE=$2
RP_DIR=$3
DOREMI_DIR=$4
OUTPUT_DIR=$5
tokenize_name_or_path=$6
max_length=$7

export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE

python preprocess.py \
	--dataset_dir ${RP_DIR} \
	--output_dir ${OUTPUT_DIR} \
        --cache_dir ${CACHE} \
	--domain $DOMAIN \
        --tokenizer $tokenize_name_or_path\
        --max_length $max_length \
