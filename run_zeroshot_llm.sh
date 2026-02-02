#!/usr/bin/env bash
# 合并脚本：对 Amazon 和 WOS 两个数据集，使用 qwen3-max, deepseek-v3, deepseek-v3.2 三个模型
# 分别跑 method1 和 method2
# 需先抽样:
#   python preprocessing/sample_amazon_by_category.py --input data/Amazon/test.json --output data/Amazon/test_sampled_1000.json --n 1000
#   python preprocessing/sample_amazon_by_category.py --input data/WOS/test.json --output data/WOS/test_sampled_1000.json --n 1000
# 需设置环境变量: DASHSCOPE_API_KEY 或 ALIYUN_LLM_API_KEY

set -e
cd "$(dirname "$0")"

SAMPLES="100,200,300,400,500,600,700,800,900,1000"
# SAMPLES="1"  # 用于快速测试

# 数据集列表
DATASETS=("Amazon" "WOS")

# 方法列表
METHODS=("one" "two")

# 模型列表
MODELS=("qwen3-max" "deepseek-v3" "deepseek-v3.2")

# 根据数据集选择输入文件（统一使用抽样集）
get_input_file() {
    local dataset=$1
    echo "data/${dataset}/test_sampled_1000.json"
}

# 根据数据集和方法生成输出文件名
get_output_file() {
    local dataset=$1
    local method=$2
    local model=$3
    # 将模型名中的点替换为下划线，避免文件名问题
    local model_safe=$(echo "$model" | tr '.' '_')
    echo "output_method${method}_${dataset}_${model_safe}_1000.json"
}

# 根据数据集和方法生成 checkpoint 文件名
get_checkpoint_file() {
    local dataset=$1
    local method=$2
    local model=$3
    local model_safe=$(echo "$model" | tr '.' '_')
    echo "ckpt_method${method}_${dataset}_${model_safe}.json"
}

echo "=========================================="
echo "开始执行：数据集 × 方法 × 模型"
echo "数据集: ${DATASETS[@]}"
echo "方法: ${METHODS[@]}"
echo "模型: ${MODELS[@]}"
echo "样本数: $SAMPLES"
echo "=========================================="
echo ""

# 三个嵌套的 for 循环
for dataset in "${DATASETS[@]}"; do
    echo ">>> 处理数据集: $dataset"

    for method in "${METHODS[@]}"; do
        method_num=$(if [ "$method" == "one" ]; then echo "1"; else echo "2"; fi)
        method_desc=$(if [ "$method" == "one" ]; then echo "full hierarchy, one call per sample"; else echo "L1 then L2 (or L3), multiple calls per sample"; fi)

        echo "  >> 方法 $method_num ($method_desc)"

        for model in "${MODELS[@]}"; do
            echo "    > 模型: $model"

            input_file=$(get_input_file "$dataset")
            output_file=$(get_output_file "$dataset" "$method" "$model")
            checkpoint_file=$(get_checkpoint_file "$dataset" "$method" "$model")

            # 检查输入文件是否存在
            if [ ! -f "$input_file" ]; then
                echo "    ⚠️  警告: 输入文件不存在: $input_file，跳过"
                continue
            fi

            # 构建命令
            cmd="python3 run_zeroshot_hierarchy.py \
                --input_sampled \"$input_file\" \
                --hierarchy data/${dataset}/label_hierarchy.txt \
                --labels data/${dataset}/labels.txt \
                --method $method \
                --samples \"$SAMPLES\" \
                --checkpoint \"$checkpoint_file\" \
                --model \"$model\" \
                --output \"$output_file\""

            # Method One 需要 concurrency 参数
            if [ "$method" == "one" ]; then
                cmd="$cmd --concurrency 1"
            fi

            # 添加 timeout
            cmd="$cmd --timeout 60"

            echo "    执行命令: $cmd"
            echo ""

            # 执行命令
            eval $cmd
			exit

            if [ $? -eq 0 ]; then
                echo "    ✓ 完成: $output_file"
            else
                echo "    ✗ 失败: $output_file"
            fi
            echo ""
        done
    done
    echo ""
done

echo "=========================================="
echo "所有任务完成！"
echo "结果文件: output_method*_*_*_1000.json"
echo "检查点文件: ckpt_method*_*_*.json"
echo "=========================================="
