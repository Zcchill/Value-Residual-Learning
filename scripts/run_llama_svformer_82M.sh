#!/bin/bash

# Path
CACHE=Value-Residual-Learning/cache
CODE_DIR=Value-Residual-Learning
ACCELERATE_PATH=

# Environment
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE
export WANDB_DIR=${CACHE}/wandb
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_OUTPUT_DIR=${CODE_DIR}/output
PREPROCESSED_DATA=${CODE_DIR}/data/processed_slimpajama_20B
TOKENIZER=${CODE_DIR}/data/tokenizer/RedPajama-INCITE-Base-7B

# Basic setting
NNODES=1
GPU_NUM=8
PORT=50176
SEED=42

# Model
MODEL_NAME=modeling_llama_svformer
NAME=modeling_llama_svformer_82M
MODEL_TYPE=llama
ATTN_TYPE=flash_attention_2
MAX_LENGTH=2048
HIDDEN_SIZE=512
INTERMEDIATE_SIZE=1792
NUM_LAYERS=8
NUM_Q_HEADS=8
NUM_KV_HEADS=8
CONFIG_OVERIDES="max_position_embeddings=${MAX_LENGTH},vocab_size=50277,hidden_size=${HIDDEN_SIZE},intermediate_size=${INTERMEDIATE_SIZE},num_hidden_layers=${NUM_LAYERS},num_attention_heads=${NUM_Q_HEADS},num_key_value_heads=${NUM_KV_HEADS},hidden_act=silu,rope_theta=10000,rms_norm_eps=1e-5,attention_dropout=0.0,tie_word_embeddings=False,eos_token_id=0,bos_token_id=0,use_cache=False"

# Data
DATA_TRAIN='{"RedPajamaCommonCrawl_length2048":0.5,"RedPajamaC4_length2048":0.2,"RedPajamaGithub_length2048":0.1,"RedPajamaStackExchange_length2048":0.05,"RedPajamaWikipedia_length2048":0.05,"RedPajamaBook_length2048":0.05,"RedPajamaArXiv_length2048":0.05}'
DATA_VALID='{"RedPajamaCommonCrawl_length2048":1.0,"RedPajamaC4_length2048":1.0,"RedPajamaGithub_length2048":1.0,"RedPajamaStackExchange_length2048":1.0,"RedPajamaWikipedia_length2048":1.0,"RedPajamaBook_length2048":1.0,"RedPajamaArXiv_length2048":1.0}'

# Optimizer
BATCH_SIZE=1024
PER_DEVICE_TRAIN_BS=32
PER_DEVICE_EVAL_BS=32
MAX_STEPS=10000
SAVE_STEPS=1000
EVAL_STEPS=1000
MAX_SAMPLES=$(($MAX_STEPS*$BATCH_SIZE))
GRADIENT_ACC_STEPS=$(($BATCH_SIZE/($NNODES*$GPU_NUM*$PER_DEVICE_TRAIN_BS)))

# Schedule
MAX_LR=6e-4
END_LR=6e-5
WARMUP_STEP=120

# Start code
${ACCELERATE_PATH}accelerate launch \
    --config_file ${CODE_DIR}/config/accelerate_deepspeed_zero2_config.yml \
    --num_machines ${NNODES} \
    --num_processes ${GPU_NUM} \
    --main_process_port ${PORT} \
    ${CODE_DIR}/src/train.py \
    --do_train \
    --bf16 \
    --modeling_name ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --config_overrides ${CONFIG_OVERIDES} \
    --domain_weight_train ${DATA_TRAIN} \
    --domain_weight_eval ${DATA_VALID} \
    --tokenizer_name ${TOKENIZER} \
    --attn_implementation ${ATTN_TYPE} \
    --dataset_dir ${PREPROCESSED_DATA} \
    --output_dir ${MODEL_OUTPUT_DIR}/${NAME} \
    --max_train_samples ${MAX_SAMPLES} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BS} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BS} \
    --max_steps ${MAX_STEPS} \
    --max_token_length ${MAX_LENGTH} \
    --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --evaluation_strategy steps \
    --eval_steps ${EVAL_STEPS} \
    --lr_scheduler_name linear_warmup_cosine \
    --learning_rate ${MAX_LR} \
    --lr_end ${END_LR} \
    --warmup_step ${WARMUP_STEP} \
    --num_warmup_stop_steps ${MAX_STEPS} \
    --optim adamw_torch \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --report_to none \
    --logging_strategy steps \
    --logging_steps 1 \
    --logging_first_step True \
    --metric_for_best_model "eval_RedPajamaCommonCrawl_length2048_loss" \
    --greater_is_better False \
    --dataset_name slimpajama \
    --remove_unused_columns True \
    --dataloader_drop_last False \
    --dataloader_num_workers 1 \
    --run_name ${NAME} \
    --seed ${SEED} \
    --cache_dir ${CACHE} \
    2>&1 | tee -a ${CODE_DIR}/logs/${NAME}.log