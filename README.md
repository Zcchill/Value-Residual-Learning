# Value Residual Learning
This repo includes instructions for running Resformer and SVformer introduced in the following [paper](https://arxiv.org/abs/2410.17897): Value Residual Learning For Alleviating  Attention Concentration In Transformers.

# Requirement
`pip install transformers=4.44.2`.

# Data
1. Download the [tokenizer](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base) and place it in the "data/tokenizer/RedPajama-INCITE-Base-7B".
2. Follow the instructions in the "README.md" located in "src_data/" to prepare "processed_slimpajama_20B" and place it in the "data/".

# Analysis
The code for entropy analysis and token similarity analysis can be found in "analyze/get_entropy.py" and "analyze/get_simlarity.py" respectively.

# Train
`mkdir logs`, `mkdir output`

Modify the "CACHE" and "CODE_DIR" in the "*.sh" file, then run `bash scripts/run_llama_baseline_82M.sh` and `bash scripts/run_llama_resformer_82M.sh`.

# Relative Loss Analysis
Run `analyze/plot_relative_loss.py`.
