#!/bin/bash

CACHE=Value-Residual-Learning/data/cache
RP_DIR=Value-Residual-Learning/data/slimpajama_20B
OUTPUT_DIR=Value-Residual-Learning/data/processed_slimpajama_20B
tokenize_name_or_path=Value-Residual-Learning/data/tokenizer/RedPajama-INCITE-Base-7B
max_length=2048

for DOMAIN in 'RedPajamaArXiv' 'RedPajamaBook' 'RedPajamaC4' 'RedPajamaCommonCrawl' 'RedPajamaGithub' 'RedPajamaStackExchange' 'RedPajamaWikipedia';
do
bash run_preprocess.sh ${DOMAIN} ${CACHE} ${RP_DIR} ${DOREMI_DIR} ${OUTPUT_DIR} ${tokenize_name_or_path} ${max_length}
done
