import os
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

def split_list_into_batches(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

def main():
    '''
    Replay a subset of the RedPajama dataset
    '''
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--data_size', type=float, default=None)
    parser.add_argument('--meta_name', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    args = parser.parse_args()
    print(args)

    # Load the data
    input_file_path = args.input_dir + '/' + args.meta_name
    all_files_train = [str(path) for path in Path(input_file_path).rglob("*") if not path.is_dir() and (str(path).split('/')[-1]).startswith('train')]
    ds = load_dataset('json',
                        data_files=all_files_train,
                        cache_dir=args.cache_dir)['train']
    meta_name = {'redpajama_set_name': args.meta_name}
    text_list = ds['text']
    random.shuffle(text_list)
    text_list = split_list_into_batches(text_list, 1000)
    print(f"{len(text_list)} examples are selected for domain {meta_name}")

    # Load tokenizer and add a separator token in the end
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="$A "+tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)])
    
    # Choose data
    text_list_select = []
    length = 0
    for text_sub_list in tqdm(text_list):
        tokenized_result = tokenizer(text_sub_list)['input_ids']
        for tokenized_x, x in zip(tokenized_result, text_sub_list):
            text_list_select.append(x)
            length += len(tokenized_x)
            if length > args.data_size:
                break
        if length > args.data_size:
            print(f"Finally chooses {length} tokens and {len(text_list_select)} examples.")
            break

    # Save data
    output_dir_domain = args.output_dir + '/' + args.meta_name
    os.makedirs(output_dir_domain, exist_ok=True)
    output_file_path = output_dir_domain + '/' + 'train.jsonl'
    with open(output_file_path, 'w') as f:
        for text in tqdm(text_list_select):
            td = {'text': text, 'meta': meta_name}
            f.write(json.dumps(td, ensure_ascii=False)+ '\n')

if __name__ == "__main__":
    main()