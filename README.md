# Dataset 
We use two datasets, Amazon and WOS, and both datasets are in the ./data folder , test_sampled_1000.json is the sampled dataset for LLM testing.

# Run RelaClass
## 1. data embedding
`python dataprocess.py --embedding_model SFR-Embedding-2_R --specific_file ./data/Amazon/train.json --columns token --dataset_save_dir xxxx`

`python dataprocess.py --embedding_model SFR-Embedding-2_R --specific_file ./data/WOS/train.json --columns token --dataset_save_dir xxxx`

## 2. progressive training
 `bash run-amazon.sh` 
 
 `bash run-webofscience.sh`

 Note: 
 * parameter `target_codebook_system_datasets` is the embedding dataset path for the current level'category prototypes using SFR-Embedding-2_R
 * parameter `ckpt_path` refers to the model obtained after the previous level of codebook training is completed. So we only use it when training the second-level and subsequent codebooks.

The trained models are saved under ckpt directory.

# Run LLM
`bash run_zeroshot_llm.sh`

## example for wos
```
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/WOS/test_sampled_1000.json \
  --hierarchy data/WOS/label_hierarchy.txt \
  --labels data/WOS/labels.txt \
  --method one \
  --samples "100,200,300,400,500,600,700,800,900,1000"\
  --concurrency 5 \
  --timeout 60 \
  --checkpoint ck_wos_method1_qwen3-max.json \
  --model qwen3-max \
  --output zeroshot_wos_method1_qwen3-max.json
```