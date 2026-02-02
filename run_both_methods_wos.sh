#!/usr/bin/env bash
# WOS 两层级：方法一和方法二都跑，最多 1000 条样本
# 需先抽样: python preprocessing/sample_amazon_by_category.py --input data/WOS/test.json --output data/WOS/test_sampled_1000.json --n 1000
# 需设置环境变量: DASHSCOPE_API_KEY 或 ALIYUN_LLM_API_KEY

set -e
cd "$(dirname "$0")"

SAMPLES="100,200,300,400,500,600,700,800,900,1000"
# SAMPLES="1"

echo "========== WOS Method One (full 2-level hierarchy, one call per sample) =========="
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/WOS/test_sampled_1000.json \
  --hierarchy data/WOS/label_hierarchy.txt \
  --labels data/WOS/labels.txt \
  --method one \
  --samples "$SAMPLES" \
  --concurrency 5 \
  --output zeroshot_wos_method1.json

echo ""
echo "========== WOS Method Two (L1 then L2, two calls per sample) =========="
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/WOS/test_sampled_1000.json \
  --hierarchy data/WOS/label_hierarchy.txt \
  --labels data/WOS/labels.txt \
  --method two \
  --samples "$SAMPLES" \
  --output zeroshot_wos_method2.json

echo ""
echo "Done. Results: zeroshot_wos_method1.json, zeroshot_wos_method2.json"
