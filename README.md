# Value Residual Learning
This repo includes instructions for running Resformer and SVformer introduced in the following [paper](https://arxiv.org/abs/2410.17897): Value Residual Learning For Alleviating  Attention Concentration In Transformers.

## Requirement
`pip install transformers=4.44.2`.

## Data
1. Download the [tokenizer](https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base) and place it in the "data/tokenizer/RedPajama-INCITE-Base-7B".
2. Follow the instructions in the "README.md" located in "src_data/" to prepare "processed_slimpajama_20B" and place it in the "data/".

## Analysis
The code for entropy analysis and token similarity analysis can be found in "analyze/get_entropy.py" and "analyze/get_simlarity.py" respectively.

## Train
`mkdir logs`, `mkdir output`

Modify the "CACHE" and "CODE_DIR" in the "*.sh" file, then run `bash scripts/run_llama_baseline_82M.sh` and `bash scripts/run_llama_resformer_82M.sh`.

## Relative Loss Analysis
Run `analyze/plot_relative_loss.py`.

## Notable attempts and variants:
1. modded nanogpt project
   - [twitter](https://x.com/Grad62304977/status/1854295837741809933)
   - [github](https://github.com/KellerJordan/modded-nanogpt)
   - [u-net form value residual](https://github.com/KellerJordan/modded-nanogpt/blob/ad8d5f820e69ecf308467276a87aba7841e4b563/records/020125_RuleTweak/eff63a8c-2f7e-4fc5-97ce-7f600dae0bc7.txt#L392)


2. rwkv7
   - [twitter](https://x.com/BlinkDL_AI/status/1857442520525119691)
   - [code](https://github.com/BlinkDL/RWKV-LM/blob/a963e4ad848426a7032fd6f7c54bdf59c11b67d7/RWKV-v7/rwkv_v7_demo.py#L276)
