# Download data
Download slimpajama data from https://huggingface.co/datasets/cerebras/SlimPajama-627B

## Re-organize data
The original data is not organized based on their domains.
Run `reorganize_data.py` to re-organize data.

## Re-select data
Sample 20B tokens data from original 627B data based on the original data proportions. Note that for Commoncrawl and C4, we only choose the first 3 and 5 chunks respectively. Run `run_selection.sh` for train split and "scp_valid.sh" for valid split.

| Data source    | Proportions | Tokens |
|----------------|-------------|--------|
| Commoncrawl    | 50%         | 10 B   |
| C4             | 20%         | 4 B    |
| GitHub         | 10%         | 2 B    |
| Books          | 5%          | 1 B    |
| ArXiv          | 5%          | 1 B    |
| Wikipedia      | 5%          | 1 B    |
| StackExchange  | 5%          | 1 B    |

## Preprocess data
For better efficiency, please tokenize the data firstly.
Run `run_all_preprocess.sh`, pass "max_length" to control the max sequence length.

