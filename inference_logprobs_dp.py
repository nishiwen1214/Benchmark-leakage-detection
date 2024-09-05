import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def find_indices(lst, value):
    indices = []
    for i, elem in enumerate(lst):
        if (elem == value and len(lst[i + 1]) != 0 and lst[i + 1][0] == ":") or elem == 'A:':
            indices.append(i)
            return indices
    return indices

def score(model, tokenizer, prompt):
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
        input_tokens = [tokenizer.decode([id]) for id in input_ids[0]]
        index = find_indices(input_tokens, 'A')
        logits = model(input_ids).logits
        all_tokens_logprobs = F.log_softmax(logits.double(), dim=2)
        input_logprobs = [all_tokens_logprobs[:, k - 1, input_ids[0, k]] for k in range(1, input_ids.shape[1])]
        input_logprobs = [input_logprobs[k].detach().cpu().numpy()[0] for k in range(len(input_logprobs))]
        del logits
        return input_tokens, input_logprobs, index[0]

def display(model, tokenizer, prompt):
    input_tokens, input_logprobs, index = score(model, tokenizer, prompt)
    all_logprobs = 0
    for i in range(index, len(input_logprobs)):
        all_logprobs = all_logprobs + input_logprobs[i]
    return all_logprobs

def main(rank, world_size, args):
    setup(rank, world_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True).to(rank)
    model = DDP(model, device_ids=[rank])

    with open(args.permutations_data_dir, 'r') as file:
        datas = json.load(file)
    
    # Split data for each process
    data_per_rank = len(datas) // world_size
    start_idx = rank * data_per_rank
    end_idx = start_idx + data_per_rank if rank != world_size - 1 else len(datas)
    local_datas = datas[start_idx:end_idx]

    local_logprobs_list = []

    for index, data in enumerate(tqdm(local_datas, desc=f"Rank {rank}")):
        result = display(model, tokenizer, data["instruction"])
        local_logprobs_list.append(result)
        if index % 1000 == 0:
            torch.cuda.empty_cache()

    # Save local results to a file
    local_save_path = os.path.join(args.save_dir, f"logprobs_rank_{rank}.json")
    with open(local_save_path, 'w') as json_file:
        json.dump(local_logprobs_list, json_file, indent=4, ensure_ascii=False)

    # cleanup()
    dist.barrier()

    # Only rank 0 will merge the files
    if rank == 0:
        logprobs_list = []
        for i in range(world_size):
            local_file_path = os.path.join(args.save_dir, f"logprobs_rank_{i}.json")
            with open(local_file_path, 'r') as json_file:
                local_logprobs = json.load(json_file)
                logprobs_list.extend(local_logprobs)
        
        # Save the merged results
        final_save_path = os.path.join(args.save_dir, "logprobs.json")
        with open(final_save_path, 'w') as json_file:
            json.dump(logprobs_list, json_file, indent=4, ensure_ascii=False)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='logprobs', description='')
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--permutations_data_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--world_size", type=int, default=8, help="number of GPUs to use")
    args = parser.parse_args()

    world_size = args.world_size
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)