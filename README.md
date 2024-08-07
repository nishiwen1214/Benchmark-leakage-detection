# Benchmark-leakage-detection

### Quickstart

Use this  command to run  inference_logprobs.py

```bash
CUDA_VISIBLE_DEVICES=0 python inference_logprobs.py --model_dir model_dir --data_dir data/Ceval.json --save_dir ave_dir 
```

Use this  command to run  get_outlier.py

```bash
python get_outlier.py --logprobs_dir model_dir --data_dir data/Ceval.json --save_dir save_dir --method IsolationForest1 --permutation_nmu 24 
```