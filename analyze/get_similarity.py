import json
import torch
from transformers import set_seed
from tqdm import tqdm
from scipy.stats import entropy
import numpy as np
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

def calculate_similarity(data, head_operation='average'):
    if head_operation == 'average':
        data = data.mean(dim=1)
    elif head_operation == 'concatenate':
        data = data.transpose(1, 2)
        data = data.reshape(data.size(0), data.size(1), -1) 
    data = data.squeeze()
    data_norm = torch.nn.functional.normalize(data, p=2, dim=-1)
    similarity_matrix = torch.matmul(data_norm, data_norm.transpose(0, 1))
    mask = torch.eye(similarity_matrix.size(1), device=similarity_matrix.device).bool()
    similarity_values = similarity_matrix[~mask].view(-1)
    avg_similarity = similarity_values.mean().item()
    return avg_similarity

def main():
    set_seed(42)
    model_name_or_path = "Meta-Llama-3.1-8B"
    data_path = "sampled_data.json"
    embedding_type = "hidden"
    head_operation = "concatenate"
    if embedding_type == "key":
        use_cache = True
        output_hidden_states = False
    elif embedding_type == "hidden":
        use_cache = False
        output_hidden_states = True
        head_operation = None
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map='auto', attn_implementation="eager", use_safetensors=True)
    model = model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    with open(data_path, "r") as json_file:
        data = json.load(json_file)

    similarity_lists = []
    for _, text in tqdm(enumerate(data), total=len(data)):

        input_ids = tokenizer.encode(text)
        input_ids = input_ids[:2048]
        input_ids = torch.tensor([input_ids]).to(model.device)
        CausalLMOutput = model(input_ids=input_ids, return_dict=False, output_hidden_states=output_hidden_states, output_attentions=False, use_cache=use_cache)
        assert len(CausalLMOutput) == 2
        similarity_list = []
        for layer_embedding in CausalLMOutput[1]:
            if embedding_type == "key":
                layer_embedding = layer_embedding[0]
            assert layer_embedding.size(0) == 1
            sim = float(calculate_similarity(layer_embedding, head_operation=head_operation))
            similarity_list.append(sim)
        similarity_lists.append(similarity_list)

    average_list = [sum(elements) / len(elements) for elements in zip(*similarity_lists)]
    print(average_list)

if __name__ == "__main__":
    main()