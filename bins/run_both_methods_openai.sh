#!/usr/bin/env bash
# 使用 OpenAI 模型运行方法一和方法二
# 需设置环境变量: OPENAI_API_KEY

set -e
cd "$(dirname "$0")"

SAMPLES="100,200,300,400,500,600,700,800,900,1000"

# OpenAI 最强模型推荐：
# - gpt-4o: 速度快，性能好，成本适中（推荐）
# - o1: 推理能力最强，但速度较慢，成本较高
# - gpt-4-turbo: 经典选择，性能与成本平衡

# 可根据需要选择以下模型之一
MODEL="gpt-4.1"  # 推荐：速度快且性能好
# MODEL="o1"      # 推理最强，适合复杂任务
# MODEL="gpt-4-turbo"  # 经典模型

echo "========== Method One with OpenAI $MODEL =========="
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method one \
  --samples "$SAMPLES" \
  --concurrency 5 \
  --timeout 60 \
  --checkpoint ck_method1_${MODEL}.json \
  --model $MODEL \
  --output zeroshot_method1_1000_${MODEL}.json

echo ""
echo "========== Method Two with OpenAI $MODEL =========="
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method two \
  --samples "$SAMPLES" \
  --timeout 60 \
  --checkpoint ck_method2_${MODEL}.json \
  --model $MODEL \
  --output zeroshot_method2_1000_${MODEL}.json

echo ""
echo "Done. Results: zeroshot_method1_1000_${MODEL}.json, zeroshot_method2_1000_${MODEL}.json"
