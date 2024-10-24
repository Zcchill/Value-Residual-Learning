import json
import jsonlines
from glob import glob
from tqdm import tqdm
import zstandard as zstd
import os

Data_Path = "Value-Residual-Learning/data/SlimPajama-627B" # Path to downloaded cerebras/SlimPajama-627B
Write_Dir = "Value-Residual-Learning/data/slimpajama_all/{}" # Path to store data from different domains
domain_list = ['RedPajamaStackExchange', 'RedPajamaArXiv', 'RedPajamaBook', 'RedPajamaC4', 'RedPajamaWikipedia', 'RedPajamaCommonCrawl', 'RedPajamaGithub']

# Train
for chunk_num in range(1,10):
    domain_examples = {k:[] for k in domain_list}
    
    paths = glob('{}/train/chunk{}/*.jsonl.zst'.format(Data_Path, chunk_num))
    for path in tqdm(paths, total=len(paths)):
        with zstd.open(path, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = json.loads(line)
                domain = line['meta']['redpajama_set_name']
                domain_examples[domain].append(line)
    print({k:len(v) for k,v in domain_examples.items()})

    for domain in domain_list:
        os.makedirs(Write_Dir.format(domain), exist_ok=True)
        write_path = Write_Dir.format(domain) + "/train_chunk{}.jsonl".format(chunk_num)
        wfp = jsonlines.open(write_path.format(domain, chunk_num), mode='w')
        wfp.write_all(domain_examples[domain])
        wfp.close()

# Validation
domain_examples = {k:[] for k in domain_list}
for chunk_num in range(1,6): 
    paths = glob('{}/validation/chunk{}/*.jsonl.zst'.format(Data_Path, chunk_num))
    for path in tqdm(paths, total=len(paths)):
        with zstd.open(path, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = json.loads(line)
                domain = line['meta']['redpajama_set_name']
                domain_examples[domain].append(line)
    print({'valid_'+k:len(v) for k,v in domain_examples.items()})

for domain in domain_list:
    os.makedirs(Write_Dir.format(domain), exist_ok=True)
    write_path = Write_Dir.format(domain) + "/validation_chunk_all.jsonl"
    wfp = jsonlines.open(write_path.format(domain), mode='w')
    wfp.write_all(domain_examples[domain])
    wfp.close()