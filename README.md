# Benchmark-leakage-detection

### Quickstart
Use this  command to run  data_process.py

```bash
python data_process.py  --data_dir data_dir --save_dir data
```

Use this  command to run  inference_logprobs.py

```bash
CUDA_VISIBLE_DEVICES=0 python inference_logprobs.py --model_dir model_dir --permutations_data_dir data/permutations_data.json --save_dir data
```

Use this  command to run  get_outlier.py

```bash
python get_outlier.py --logprobs_dir data/logprobs.json --permutations_data_dir data/permutations_data.json --save_dir data --method IsolationForest --permutation_nmu 24
```