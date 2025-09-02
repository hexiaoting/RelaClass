# Dataset 
We use two datasets, Amazon and WOS, and both datasets are in the ./data folder 

# Run RelaClass
# 1. data embedding
`python dataprocess.py --embedding_model SFR-Embedding-2_R --specific_file ./data/Amazon/train.json --columns token --dataset_save_dir xxxx`

`python dataprocess.py --embedding_model SFR-Embedding-2_R --specific_file ./data/WOS/train.json --columns token --dataset_save_dir xxxx`

# 2. progressive training
see file `run-amazon.sh` and `run-webofscience.sh`
