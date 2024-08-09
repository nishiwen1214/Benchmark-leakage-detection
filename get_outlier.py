from sklearn.ensemble import IsolationForest
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(prog='get_outlier', description='')
parser.add_argument("--logprobs_dir", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--permutation_nmu", type=int)
args = parser.parse_args()
thresholds = [-0.2, -0.17, -0.15]

with open(args.data_dir, 'r') as file:
    list_data = json.load(file)
with open(args.logprobs_dir, 'r') as file:
    list_logprobs = json.load(file)

list_data = [list_data[i:i + args.permutation_nmu] for i in range(0, len(list_data), args.permutation_nmu)]
list_logprobs = [list_logprobs[i:i + args.permutation_nmu] for i in range(0, len(list_logprobs), args.permutation_nmu)]

if args.method == "IsolationForest":
    outliers = [[], [], []]
    for index, data in enumerate(list_logprobs):
        X = np.array(data).reshape(-1, 1)
        clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        clf.fit(X)
        scores = clf.decision_function(X)
        max_value_index = np.argmax(data)
        max_value_score = scores[max_value_index]
        for outlier_index, threshold in enumerate(thresholds):
            if max_value_score < threshold:
                outlier = {
                    "index": str(index),
                    "max_value_index": str(max_value_index),
                    "data": list_data[index][max_value_index]["instruction"],
                    "logprobs": data[max_value_index]
                }
                outliers[outlier_index].append(outlier)

    for i, threshold in enumerate(thresholds):
        print(f"模型阈值{threshold},数据泄露百分比为{len(outliers[i]) / len(list_data):.2f}%")
        with open(f'{args.save_dir}/outliers{threshold}.json', 'w') as json_file:
            json.dump(outliers[i], json_file, indent=4, ensure_ascii=False)
else:
    outliers = []
    for index, data in enumerate(list_logprobs):
        max = data[0]
        isMax = True
        for temp in data[1:]:
            if temp > max:
                isMax = False
                break
        if isMax:
            dict = {
                "index": str(index),
                "max_value_index": str(0),
                "data": list_data[index][0]["instruction"],
                "logprobs": data[0]
            }
            outliers.append(dict)
    with open(f'{args.save_dir}/outliers_max.json', 'w') as json_file:
        print(f"模型数据泄露百分比为{len(outliers) / len(list_data):.2f}%")
        json.dump(outliers, json_file, indent=4, ensure_ascii=False)
