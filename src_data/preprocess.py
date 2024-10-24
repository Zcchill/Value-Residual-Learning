import os
import argparse
from pathlib import Path
import shutil
from itertools import chain
from datasets import load_dataset
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

def get_transform(tokenizer, max_length, domain_id, seed=None):
    def transform(batch):

        # Concatenate all texts.
        examples = tokenizer(batch['text'])
        examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(examples[list(examples.keys())[0]])
        
        # Drop the small remainder
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in examples.items()
        }
        result['domain_id'] = [domain_id for _ in range(0, total_length, max_length)]
        return result

    return transform


def main():
    '''
    Preprocess a subset of the RedPajama dataset
    '''
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/path/to/redpajama')
    parser.add_argument('--output_dir', type=str, default='/path/to/output_dir')
    parser.add_argument('--domain', type=str, default='common_crawl')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--tokenizer', type=str, default='/path/to/tokenizer')
    parser.add_argument('--cache_dir', type=str, default='/path/to/cache')
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    os.makedirs(args.output_dir+'/'+'train', exist_ok=True)
    os.makedirs(args.output_dir+'/'+'validation', exist_ok=True)
    output_dir_train = Path(args.output_dir) / 'train' / f"{args.domain}_length{args.max_length}"
    print(output_dir_train)
    output_dir_val = Path(args.output_dir) / 'validation' / f"{args.domain}_length{args.max_length}"
    if output_dir_train.exists() and output_dir_val.exists():
        print("Already done, skipping")
        return

    # Figure out data files.
    dataset_dir = Path(args.dataset_dir)
    DOMAINS = list(sorted([str(domain_dir.name) for domain_dir in dataset_dir.iterdir() if not str(domain_dir.name).endswith('txt')]))
    DOMAIN_TO_IDX = {
        name: idx for idx, name in enumerate(DOMAINS)}
    assert(args.domain in DOMAINS)
    domain_dir = dataset_dir / args.domain
    all_files_train = [str(path) for path in domain_dir.rglob("*") if not path.is_dir() and (str(path).split('/')[-1]).startswith('train')]
    all_files_val = [str(path) for path in domain_dir.rglob("*") if not path.is_dir() and (str(path).split('/')[-1]).startswith('validation')]
    
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f'eos_token is {tokenizer.eos_token} and eos_token_id is {tokenizer.eos_token_id}')
    tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="$A "+tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)])

    # Load data.
    ds_val = load_dataset('json',
                        data_files=all_files_val,
                        cache_dir=args.cache_dir,
                        num_proc=args.nproc)['train']
    ds_train = load_dataset('json',
                        data_files=all_files_train,
                        cache_dir=args.cache_dir,
                        num_proc=args.nproc)['train']

    # Tokenize the data.
    transform = get_transform(tokenizer, args.max_length, DOMAIN_TO_IDX[args.domain], seed=args.seed)
    ds_train_transform = ds_train.map(transform, batched=True, remove_columns=ds_train.column_names)
    ds_val_transform = ds_val.map(transform, batched=True, remove_columns=ds_val.column_names)
    print(f'The ds_val_transform: {ds_val_transform}')
    print(f'The ds_train_transform: {ds_train_transform}')
    print(f'Example of ds_val_transform: {ds_val_transform[0]}')

    # Save to disk
    output_dir_val.mkdir(exist_ok=True)
    ds_val_transform.save_to_disk(str(output_dir_val), max_shard_size='1GB')
    output_dir_train.mkdir(exist_ok=True)
    ds_train_transform.save_to_disk(str(output_dir_train), max_shard_size='1GB')

    # del some of the cache
    shutil.rmtree(str(Path(args.cache_dir) / 'json'))
    ds_val_transform.cleanup_cache_files()
    ds_val.cleanup_cache_files()
    ds_train_transform.cleanup_cache_files()
    ds_train.cleanup_cache_files()

if __name__ == '__main__':
    main()