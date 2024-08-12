import json
import itertools
import argparse

"""
{
   'option': {
   'A': '由间充质增生形成', 
   'B': '人胚第4周出现', 
   'C': '相邻鳃弓之间为鳃沟',
    'D': '共5对鳃弓'
    },
   'question': '下列有关鳃弓的描述，错误的是'
}
"""
parser = argparse.ArgumentParser(prog='data_process', description='')
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()
with open(args.data_dir, 'r') as file:
    data_list = json.load(file)

# 定义你的字符列表
chars = ['A', 'B', 'C', 'D']

# 使用itertools.permutations生成所有排列yy
permutations_list = list(itertools.permutations(chars))
result = []

for index, row in enumerate(data_list):

    for perm in permutations_list:
        instruction = {
            "instruction":
f"""
{row['question']}:
A:{row['option'][perm[0]]}
B:{row["option"][perm[1]]}
C:{row["option"][perm[2]]} 
D:{row["option"][perm[3]]}
""",
            # "instruction": row['question'] + "\n" + "A:" + row["option"][perm[0]] + "\n" + "B:" + row["option"][
            #     perm[1]] + "\n" + "C:" + row["option"][perm[2]] + "\n" + "D:" + row["option"][perm[3]] + "\n",
        }
        result.append(instruction)

with open(f"{args.save_dir}/permutations_data.json", 'w') as json_file:
    json.dump(result, json_file, indent=4, ensure_ascii=False)
