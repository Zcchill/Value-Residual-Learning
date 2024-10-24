#!/bin/bash

CACHE=Value-Residual-Learning/cache
RP_DIR=Value-Residual-Learning/data/slimpajama_all
OUTPUT_DIR=Value-Residual-Learning/data/slimpajama_20B
tokenize_name_or_path=Value-Residual-Learning/data/tokenizer/RedPajama-INCITE-Base-7B

DOMAINS=('RedPajamaGithub' 'RedPajamaBook' 'RedPajamaArXiv' 'RedPajamaWikipedia' 'RedPajamaStackExchange' 'RedPajamaCommonCrawl' 'RedPajamaC4')
DATA_SIZES=('2000000000' '1000000000' '1000000000' '1000000000' '1000000000' '10000000000' '4000000000')

for i in "${!DOMAINS[@]}"; do
    DOMAIN=${DOMAINS[$i]}
    DATA_SIZE=${DATA_SIZES[$i]}

    python selection.py \
        --input_dir ${RP_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --data_size ${DATA_SIZE} \
        --meta_name $DOMAIN \
        --tokenizer $tokenize_name_or_path \
        --cache_dir ${CACHE}
done
