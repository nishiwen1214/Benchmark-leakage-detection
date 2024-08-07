from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import json
import tqdm
import argparse
parser = argparse.ArgumentParser(prog='logprobs', description='')
parser.add_argument("--model_dir", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto", trust_remote_code=True,
                                             attn_implementation="flash_attention_2", torch_dtype="auto").eval()


def find_indices(lst, value):
    indices = []
    for i, elem in enumerate(lst):
        if (elem == value and len(lst[i + 1]) != 0 and lst[i + 1][0] == ":") or elem == 'A:':
            indices.append(i)
            return indices
    return indices


def score(prompt):
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


def display(prompt):
    input_tokens, input_logprobs, index = score(prompt)
    all_logprobs = 0
    for i in range(index, len(input_logprobs)):
        all_logprobs = all_logprobs + input_logprobs[i]
    return all_logprobs


with open(args.data_dir, 'r') as file:
    datas = json.load(file)
logprobs_list = []
i = 0
for data in tqdm.tqdm(datas):

    result = display(data["instruction"])
    logprobs_list.append(result)

    if i % 1000 == 0:
        torch.cuda.empty_cache()
    i = i + 1

with open(f"{args.save_dir}/logprobs.json", 'w') as json_file:
    json.dump(logprobs_list, json_file, indent=4, ensure_ascii=False)
