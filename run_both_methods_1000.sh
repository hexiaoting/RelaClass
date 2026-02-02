#!/usr/bin/env bash
# 方法一和方法二都跑，最多 1000 条样本，每 100 条统计一次（100,200,...,1000）
# 需设置环境变量: DASHSCOPE_API_KEY 或 ALIYUN_LLM_API_KEY

set -e
cd "$(dirname "$0")"

SAMPLES="100,200,300,400,500,600,700,800,900,1000"

echo "========== Method One (full hierarchy, one call per sample) =========="
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method one \
  --samples "$SAMPLES" \
  --concurrency 5 \
  --model qwen3-max \
  --output zeroshot_method1_1000_qwen3-max.json

python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method one \
  --samples "$SAMPLES" \
  --concurrency 5 \
  --model deepseek-v3.2 \
  --output zeroshot_method1_1000_deepseek-v3.2.json
exit 0
echo ""
echo "========== Method Two (L1 then L2 then L3, three calls per sample) =========="
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method two \
  --samples "$SAMPLES" \
  --model qwen3-max \
  --output zeroshot_method2_1000_qwen3-max.json

echo ""
echo "Done. Results: zeroshot_method1_1000.json, zeroshot_method2_1000.json"
