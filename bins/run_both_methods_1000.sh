#!/usr/bin/env bash
# 方法一和方法二都跑，最多 1000 条样本，每 100 条统计一次（100,200,...,1000）
# 需设置环境变量: DASHSCOPE_API_KEY 或 ALIYUN_LLM_API_KEY

set -e
cd "$(dirname "$0")"

SAMPLES="1000,2000,3000,4000,5000,6000,7000,8000,9000,10000"

echo "========== Method One (full hierarchy, one call per sample) =========="
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method one \
  --samples "$SAMPLES" \
  --concurrency 5 \
  --timeout 60 \
  --checkpoint ckpt_method1_amazon_1000_qwen3-max.json \
  --model qwen3-max \
  --output zeroshot_method1_amazon_1000_qwen3-max.json

python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method one \
  --samples "$SAMPLES" \
  --concurrency 5 \
  --timeout 60 \
  --checkpoint ckpt_method1_amazon_1000_deepseek-v3.2.json \
  --model deepseek-v3.2 \
  --output zeroshot_method1_amazon_1000_deepseek-v3.2.json



echo "========== Method Two (L1 then L2 then L3, three calls per sample) =========="
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method two \
  --samples "$SAMPLES" \
  --timeout 60 \
  --checkpoint ckpt_method2_amazon_1000_qwen3-max.json \
  --model qwen3-max \
  --output zeroshot_method2_amazon_1000_qwen3-max.json

python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method two \
  --samples "$SAMPLES" \
  --timeout 60 \
  --checkpoint ckpt_method2_amazon_1000_deepseek-v3.2.json \
  --model deepseek-v3.2 \
  --output zeroshot_method2_amazon_1000_deepseek-v3.2.json

echo ""
echo "Done. Results: zeroshot_method1_amazon_1000_*.json, zeroshot_method2_amazon_1000_*.json"
