
### 2. 使用预设脚本（推荐）

直接运行 OpenAI 专用脚本：

```bash
./run_both_methods_openai.sh
```

默认使用 `gpt-4o` 模型。如需使用其他模型，编辑脚本中的 `MODEL` 变量：

```bash
MODEL="gpt-4o"      # 推荐：速度快且性能好
# MODEL="o1"        # 推理最强
# MODEL="gpt-4-turbo"  # 经典模型
```

### 3. 手动运行

你也可以直接使用 Python 脚本：

```bash
# 方法一 - 使用 gpt-4o
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method one \
  --model gpt-4o \
  --output results_gpt4o.json

# 方法二 - 使用 o1
python3 run_zeroshot_hierarchy.py \
  --input_sampled data/Amazon/test_sampled_1000.json \
  --hierarchy data/Amazon/label_hierarchy.txt \
  --labels data/Amazon/labels.txt \
  --method two \
  --model o1 \
  --output results_o1.json
```


### 阿里云模型（原有）
- `deepseek-v3`
- `deepseek-v3.2`
- `qwen3-max`

需设置环境变量：`DASHSCOPE_API_KEY` 或 `ALIYUN_LLM_API_KEY`
