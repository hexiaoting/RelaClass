"""
Zero-shot 层级分类：读抽样结果 + hierarchy + labels，
支持方法一（全体系单次）与方法二（逐级 L1→L2→L3），
按 100/500/1000 样本统计耗时与层级一致准确率。使用阿里云 DeepSeek v3。
支持 timeout、重试、断点续跑。
"""
import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


# API：必须通过环境变量配置，避免在代码中硬编码密钥
def _get_client(model_name):
    """根据模型名称返回对应的 OpenAI 客户端"""
    # OpenAI 模型列表
    openai_models = ["gpt-5.2", "o3", "o1", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4.1"]

    if model_name in openai_models:
        # 使用 OpenAI API
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not set. Set OPENAI_API_KEY in the environment."
            )
        return OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1")
    else:
        # 使用阿里云 API (DeepSeek/Qwen)
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIYUN_LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "API key not set. Set DASHSCOPE_API_KEY or ALIYUN_LLM_API_KEY in the environment."
            )
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


MODEL_DEFAULT = "deepseek-v3"
MODEL_CHOICES = ["deepseek-v3", "deepseek-v3.2", "qwen3-max", "o3", "o1", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-5.2", "gpt-4.1"]
client = None  # 延迟初始化，根据具体模型创建

# 全局配置：由 parse_args 与 run_eval 传入，供 API 调用使用
_timeout_s = 120
_max_retries = 3


def _create_completion_with_retry(timeout=None, max_retries=None, model=None, **kwargs):
    """带超时与重试的 API 调用。失败时指数退避重试。"""
    timeout = timeout if timeout is not None else _timeout_s
    max_retries = max_retries if max_retries is not None else _max_retries

    # 根据模型获取对应的 client
    model_name = model or kwargs.get("model") or MODEL_DEFAULT
    client_for_model = _get_client(model_name)

    last_error = None
    for attempt in range(max_retries):
        try:
            # 确保 model 参数被传递给 API 调用
            return client_for_model.chat.completions.create(model=model_name, **kwargs, timeout=timeout)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2**attempt  # 1s, 2s, 4s
                print(f"[API] 请求失败 (attempt {attempt + 1}/{max_retries}): {e}，{wait}s 后重试...")
                time.sleep(wait)
    raise last_error


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot hierarchy classification")
    p.add_argument("--input_sampled", type=str, default="data/Amazon/test_sampled_1000.json")
    p.add_argument("--hierarchy", type=str, default="data/Amazon/label_hierarchy.txt")
    p.add_argument("--labels", type=str, default="data/Amazon/labels.txt")
    p.add_argument("--method", type=str, choices=["one", "two"], default="one")
    p.add_argument("--model", type=str, choices=MODEL_CHOICES, default=MODEL_DEFAULT,
                   help=f"Model to use. Choices: {MODEL_CHOICES}")
    p.add_argument("--samples", type=str, default="100,500,1000",
                   help="Comma-separated sample sizes for reporting, e.g. 100,500,1000")
    p.add_argument("--output", type=str, default="zeroshot_results.json")
    p.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    p.add_argument("--debug", action="store_true", help="Print model input/output and eval per sample")
    p.add_argument("--timeout", type=int, default=120, help="API request timeout in seconds")
    p.add_argument("--max_retries", type=int, default=3, help="Max retries on API failure")
    p.add_argument("--checkpoint", type=str, default="", help="Checkpoint file path for resume. If exists, loads and skips completed samples; saves progress periodically.")
    return p.parse_args()


# ---------- 分类体系加载（hierarchy 根 0→top、id2name 支持 Amazon 整名）----------

def load_id2name(labels_path):
    """labels.txt: id\\tname，名称整体使用（不按 '-' 切分）。"""
    id2name = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    lid = int(parts[0])
                    name = parts[1].strip()
                    id2name[lid] = name
                except ValueError:
                    pass
    return id2name


def load_hierarchy(hierarchy_path):
    """parent\\tchild 每行。构建 parent->[children]，并设 hierarchy['top'] = 根节点列表（从未作为 child 出现的节点）。"""
    hierarchy = defaultdict(list)
    all_children = set()
    with open(hierarchy_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                p, c = parts[0], parts[1]
                hierarchy[p].append(c)
                all_children.add(c)
    roots = sorted([k for k in hierarchy.keys() if k not in all_children], key=int)
    hierarchy["top"] = roots
    return hierarchy


def _normalize_name(name):
    if not name:
        return ""
    s = name.strip().replace(" ", "_").replace("&", "and").lower()
    return s


def resolve_pred_name_to_id(pred_name, id2name, candidate_ids=None):
    """将预测名称解析为 id。candidate_ids 为可选的一级/二级/三级候选 id 列表。当提供 candidate_ids 时，仅从候选中返回，禁止 fuzzy_global 跨候选匹配。"""
    norm_pred = _normalize_name(pred_name)
    name_to_id = {_normalize_name(v): k for k, v in id2name.items()}
    if candidate_ids is not None:
        # 有候选时仅从候选中解析，避免误匹配到其他层级的同名/相似类
        for cid in candidate_ids:
            cname = _normalize_name(id2name.get(cid, ""))
            if cname == norm_pred:
                return cid
            if norm_pred in cname or cname in norm_pred:
                return cid
        return None
    if norm_pred in name_to_id:
        return name_to_id[norm_pred]
    for name, lid in name_to_id.items():
        if norm_pred in name or name in norm_pred:
            return lid
    return None


# ---------- 方法一：全体系单次 prompt ----------

def get_instruct_full(id2name, hierarchy, num_levels=3):
    """Full hierarchy (2 or 3 levels) + instruction to return category names (semicolon-separated)."""
    top_ids = hierarchy["top"]
    l1_names = [id2name[int(i)] for i in top_ids]
    lines = [
        f"You are a hierarchical text classifier. The first level has {len(l1_names)} categories: {', '.join(l1_names)}.",
        "",
    ]
    for l1_id in top_ids:
        l1_id = str(l1_id)
        if l1_id not in hierarchy:
            continue
        l2_ids = hierarchy[l1_id]
        l2_names = [id2name[int(i)] for i in l2_ids]
        lines.append(f"Under {id2name[int(l1_id)]}, the subcategories are: {', '.join(l2_names)}.")
        if num_levels == 3:
            for l2_id in l2_ids:
                l2_id = str(l2_id)
                if l2_id not in hierarchy:
                    continue
                l3_ids = hierarchy[l2_id]
                l3_names = [id2name[int(i)] for i in l3_ids]
                lines.append(f"  - Under {id2name[int(l2_id)]}, the subcategories are: {', '.join(l3_names)}.")
    lines.append("")
    if num_levels == 2:
        lines.append("Classify the following product description into this two-level hierarchy. Return only the two category names separated by semicolons, with no other explanation. Example format: Category1;Category2")
    else:
        lines.append("Classify the following product description into this three-level hierarchy. Return only the three category names separated by semicolons, with no other explanation. Example format: Category1;Category2;Category3")
    return "\n".join(lines)


def _strip_category_prefix(s):
    """Strip leading 'Category N:' or 'Level N:' from model output for matching."""
    return re.sub(r"^(Category|Level)\s*\d*\s*:\s*", "", s, flags=re.I).strip()


def _extract_single_category(content, id2name, candidate_ids):
    """从方法二逐级调用的模型输出中提取单个类名。优先提取 **X**、逐行尝试解析，避免仅用首行导致误解析。"""
    if not content or not content.strip():
        return None
    text = content.strip()
    # 1. 提取 **CategoryName** 模式
    bold_matches = re.findall(r"\*\*([^*]+)\*\*", text)
    for m in bold_matches:
        cleaned = m.strip()
        if cleaned and len(cleaned) < 80:
            rid = resolve_pred_name_to_id(cleaned, id2name, candidate_ids=candidate_ids)
            if rid is not None:
                return rid
    # 2. 逐行尝试，优先能解析为首选候选的行
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in lines:
        raw = _strip_category_prefix(ln)
        if not raw or len(raw) > 100:
            continue
        rid = resolve_pred_name_to_id(raw, id2name, candidate_ids=candidate_ids)
        if rid is not None:
            return rid
    return None


def parse_method_one_response(text, num_levels=3):
    """Parse two or three category names from model response (semicolon or newline separated)."""
    text = text.strip()
    if not text:
        return (None, None, None) if num_levels == 3 else (None, None, None)
    for sep in ["\n", "；", ";"]:
        if sep in text:
            parts = [p.strip() for p in text.replace("；", ";").split(sep)]
            parts = [p for p in parts if p and not p.startswith("类别") and len(p) < 100]
            if num_levels == 2:
                if len(parts) >= 2:
                    return _strip_category_prefix(parts[0]), _strip_category_prefix(parts[1]), None
                if len(parts) == 1 and ";" in parts[0]:
                    parts = [p.strip() for p in parts[0].split(";")]
                if len(parts) >= 2:
                    return _strip_category_prefix(parts[0]), _strip_category_prefix(parts[1]), None
            else:
                if len(parts) >= 3:
                    return _strip_category_prefix(parts[0]), _strip_category_prefix(parts[1]), _strip_category_prefix(parts[2])
                if len(parts) == 1 and ";" in parts[0]:
                    parts = [p.strip() for p in parts[0].split(";")]
                if len(parts) >= 3:
                    return _strip_category_prefix(parts[0]), _strip_category_prefix(parts[1]), _strip_category_prefix(parts[2])
    parts = [p.strip() for p in re.split(r"[;；\n]", text)]
    parts = [p for p in parts if p][:3]
    if num_levels == 2 and len(parts) >= 2:
        return _strip_category_prefix(parts[0]), _strip_category_prefix(parts[1]), None
    if num_levels == 3 and len(parts) >= 3:
        return _strip_category_prefix(parts[0]), _strip_category_prefix(parts[1]), _strip_category_prefix(parts[2])
    return None, None, None


def _truncate(s, max_len=200):
    """Truncate string for debug print."""
    s = str(s)
    return s[:max_len] + "..." if len(s) > max_len else s


def load_checkpoint(checkpoint_path):
    """加载断点文件，返回 (results_by_idx: dict, method, model) 或 (None, None, None)。"""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return None, None, None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results_raw = data.get("results", {})
        results_by_idx = {int(k): v for k, v in results_raw.items()}
        return results_by_idx, data.get("method"), data.get("model")
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint {checkpoint_path}: {e}")
        return None, None, None


def _is_request_success(result):
    """请求成功（API 有返回）则 True；quota/网络等导致未消耗 token 则 False，不应写入 checkpoint。"""
    usage = result.get("usage") or {}
    return (usage.get("total_tokens") or 0) > 0


def save_checkpoint(checkpoint_path, results_by_idx, method, model):
    """保存断点到 JSON 文件。仅写入请求成功的样本（usage.total_tokens > 0），失败（如 quota）的不写入以便续跑时重试。"""
    if not checkpoint_path:
        return
    try:
        successful = {k: v for k, v in results_by_idx.items() if _is_request_success(v)}
        data = {
            "method": method,
            "model": model,
            "results": {str(k): v for k, v in successful.items()},
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint {checkpoint_path}: {e}")


def _get_usage(completion):
    """Extract token usage from completion.usage. Returns dict with prompt_tokens, completion_tokens, total_tokens or zeros."""
    u = getattr(completion, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }


def _sum_usage(*usage_dicts):
    """Sum multiple usage dicts."""
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage_dicts:
        if u:
            out["prompt_tokens"] += u.get("prompt_tokens", 0)
            out["completion_tokens"] += u.get("completion_tokens", 0)
            out["total_tokens"] += u.get("total_tokens", 0)
    return out


def classify_method_one(token, system_prompt, id2name, hierarchy, num_levels=3, debug=False, model=None):
    """单条样本方法一：一次调用，返回 (pred_l1_id, pred_l2_id, pred_l3_id) 或 (None,)*3。"""
    model = model or MODEL_DEFAULT
    user_content = "Product description:\n" + token
    if debug:
        print("[DEBUG method_one] === Model input ===")
        print("[DEBUG] system (first 500 chars):", _truncate(system_prompt, 500))
        print("[DEBUG] user:", _truncate(user_content, 400))
    try:
        t0 = time.time()
        completion = _create_completion_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        request_time_s = time.time() - t0
        usage = _get_usage(completion)
        content = (completion.choices[0].message.content or "").strip()
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        if debug:
            print("[DEBUG] model raw output:", _truncate(content, 300))
        n1, n2, n3 = parse_method_one_response(content, num_levels=num_levels)
        if debug:
            print("[DEBUG] parsed L1/L2/L3 names:", (n1, n2, n3))
        if n1 is None:
            return (None, None, None), usage, content, request_time_s
        top_ids = [str(x) for x in hierarchy["top"]]
        pred_l1 = resolve_pred_name_to_id(n1, id2name, candidate_ids=[int(x) for x in top_ids])
        if pred_l1 is None:
            return (None, None, None), usage, content, request_time_s
        l2_ids = hierarchy.get(str(pred_l1), [])
        pred_l2 = resolve_pred_name_to_id(n2, id2name, candidate_ids=[int(x) for x in l2_ids])
        if pred_l2 is None:
            return (pred_l1, None, None), usage, content, request_time_s
        if num_levels == 2:
            if debug:
                print("[DEBUG] pred ids (L1,L2):", (pred_l1, pred_l2))
            return (pred_l1, pred_l2, None), usage, content, request_time_s
        l3_ids = hierarchy.get(str(pred_l2), [])
        pred_l3 = resolve_pred_name_to_id(n3, id2name, candidate_ids=[int(x) for x in l3_ids])
        # 若 L3 在 predicted L2 的子树中未匹配，但名称是合法 label，则做全局回退解析（如 Drum_&_Percussion_Accessories 在 83 下无匹配，但全局对应 108）
        if pred_l3 is None and n3 and n3.strip():
            pred_l3 = resolve_pred_name_to_id(n3, id2name, candidate_ids=None)
        if debug:
            print("[DEBUG] pred ids (L1,L2,L3):", (pred_l1, pred_l2, pred_l3))
        return (pred_l1, pred_l2, pred_l3), usage, content, request_time_s
    except Exception as e:
        print(f"API error: {e}")
        return (None, None, None), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "", 0.0


# ---------- 方法二：逐级 L1 → L2 → L3 三次调用 ----------

def classify_l1_only(token, id2name, hierarchy, debug=False, model=None):
    """Choose L1 only: returns (pred_l1_id, usage_dict)."""
    model = model or MODEL_DEFAULT
    top_ids = hierarchy["top"]
    l1_names = [id2name[int(i)] for i in top_ids]
    prompt = f"From the following categories, choose the one that best matches the product description below.\nCategories: {', '.join(l1_names)}\n\nProduct description:\n{token}\n\nReturn only the category name, no explanation."
    if debug:
        print("[DEBUG L1] === Input ===")
        print("[DEBUG L1]", _truncate(prompt, 400))
    try:
        t0 = time.time()
        completion = _create_completion_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        request_time_s = time.time() - t0
        usage = _get_usage(completion)
        content = (completion.choices[0].message.content or "").strip()
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        if debug:
            print("[DEBUG L1] model output:", _truncate(content, 200))
        pred = _extract_single_category(content, id2name, candidate_ids=[int(x) for x in top_ids])
        if debug:
            print("[DEBUG L1] pred_id:", pred)
        return pred, usage, content, prompt, request_time_s
    except Exception as e:
        print(f"API L1 error: {e}")
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "", "", 0.0


def classify_l2_only(token, pred_l1_id, id2name, hierarchy, debug=False, model=None):
    """Choose L2 given L1. Returns (pred_l2_id, usage_dict)."""
    model = model or MODEL_DEFAULT
    l2_ids = hierarchy.get(str(pred_l1_id), [])
    if not l2_ids:
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "", "", 0.0
    l2_names = [id2name[int(i)] for i in l2_ids]
    prompt = f"From the following subcategories, choose the one that best matches the product description below.\nSubcategories: {', '.join(l2_names)}\n\nProduct description:\n{token}\n\nReturn only the category name, no explanation."
    if debug:
        print("[DEBUG L2] === Input ===")
        print("[DEBUG L2]", _truncate(prompt, 400))
    try:
        t0 = time.time()
        completion = _create_completion_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        request_time_s = time.time() - t0
        usage = _get_usage(completion)
        content = (completion.choices[0].message.content or "").strip()
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        if debug:
            print("[DEBUG L2] model output:", _truncate(content, 200))
        pred = _extract_single_category(content, id2name, candidate_ids=[int(x) for x in l2_ids])
        if debug:
            print("[DEBUG L2] pred_id:", pred)
        return pred, usage, content, prompt, request_time_s
    except Exception as e:
        print(f"API L2 error: {e}")
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "", "", 0.0


def classify_l3_only(token, pred_l2_id, id2name, hierarchy, debug=False, model=None):
    """Choose L3 given L2. Returns (pred_l3_id, usage_dict)."""
    model = model or MODEL_DEFAULT
    l3_ids = hierarchy.get(str(pred_l2_id), [])
    if not l3_ids:
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "", "", 0.0
    l3_names = [id2name[int(i)] for i in l3_ids]
    prompt = f"From the following subcategories, choose the one that best matches the product description below.\nSubcategories: {', '.join(l3_names)}\n\nProduct description:\n{token}\n\nReturn only the category name, no explanation."
    if debug:
        print("[DEBUG L3] === Input ===")
        print("[DEBUG L3]", _truncate(prompt, 400))
    try:
        t0 = time.time()
        completion = _create_completion_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        request_time_s = time.time() - t0
        usage = _get_usage(completion)
        content = (completion.choices[0].message.content or "").strip()
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        if debug:
            print("[DEBUG L3] model output:", _truncate(content, 200))
        pred = _extract_single_category(content, id2name, candidate_ids=[int(x) for x in l3_ids])
        if debug:
            print("[DEBUG L3] pred_id:", pred)
        return pred, usage, content, prompt, request_time_s
    except Exception as e:
        print(f"API L3 error: {e}")
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "", "", 0.0


def classify_method_two(token, id2name, hierarchy, num_levels=3, debug=False, model=None):
    """单条样本方法二：2 层时两次调用 L1→L2，3 层时三次调用 L1→L2→L3。"""
    raw = {}
    prompt_out = {}
    p1, u1, raw["l1"], prompt_out["l1"], t1 = classify_l1_only(token, id2name, hierarchy, debug=debug, model=model)
    if p1 is None:
        raw["l2"] = raw["l3"] = prompt_out["l2"] = prompt_out["l3"] = None
        return (None, None, None), u1, {"raw_output": raw, "prompt": prompt_out, "request_time_s": t1}
    p2, u2, raw["l2"], prompt_out["l2"], t2 = classify_l2_only(token, p1, id2name, hierarchy, debug=debug, model=model)
    if p2 is None:
        raw["l3"] = prompt_out["l3"] = None
        return (p1, None, None), _sum_usage(u1, u2), {"raw_output": raw, "prompt": prompt_out, "request_time_s": t1 + t2}
    if num_levels == 2:
        raw["l3"] = None
        prompt_out["l3"] = None
        return (p1, p2, None), _sum_usage(u1, u2), {"raw_output": raw, "prompt": prompt_out, "request_time_s": t1 + t2}
    p3, u3, raw["l3"], prompt_out["l3"], t3 = classify_l3_only(token, p2, id2name, hierarchy, debug=debug, model=model)
    return (p1, p2, p3), _sum_usage(u1, u2, u3), {"raw_output": raw, "prompt": prompt_out, "request_time_s": t1 + t2 + t3}


# ---------- 准确率（层级一致）与统计 ----------

def hierarchical_correct(gold, pred, num_levels=3):
    """gold/pred 为 (l1,l2,l3) 或 (l1,l2,None)。返回 (l1_ok, l2_ok, l3_ok) 层级一致正确；2 层时 l3_ok=l2_ok。"""
    g1, g2, g3 = gold[0], gold[1], gold[2]
    p1, p2, p3 = pred[0], pred[1], pred[2]
    l1_ok = (p1 is not None and p1 == g1)
    l2_ok = l1_ok and (p2 is not None and p2 == g2)
    if num_levels == 2:
        l3_ok = l2_ok
    else:
        l3_ok = l2_ok and (p3 is not None and p3 == g3)
    return l1_ok, l2_ok, l3_ok


def _classify_one_sample(args):
    """单条样本分类，供并行调用。返回 (index, result_dict)。"""
    i, obj, method, id2name, hierarchy, num_levels, system_prompt_one, debug, model = args
    token = obj.get("token", "")
    raw = obj["labels"]
    gold = (raw[0], raw[1], None) if len(raw) == 2 else tuple(raw)
    raw_output, method_two_out = None, None
    if method == "one":
        pred, usage, raw_output, request_time_s = classify_method_one(token, system_prompt_one, id2name, hierarchy, num_levels=num_levels, debug=debug, model=model)
    else:
        pred, usage, method_two_out = classify_method_two(token, id2name, hierarchy, num_levels=num_levels, debug=debug, model=model)
        request_time_s = method_two_out.get("request_time_s", 0.0) if method_two_out else 0.0
    pred = (pred[0], pred[1], pred[2])
    l1_ok, l2_ok, l3_ok = hierarchical_correct(gold, pred, num_levels=num_levels)
    if method == "one":
        result = {
            "gold": gold, "pred": pred, "l1_ok": l1_ok, "l2_ok": l2_ok, "l3_ok": l3_ok, "usage": usage,
            "raw_output": raw_output,
            "user_prompt": "Product description:\n" + token,
            "time_s": request_time_s,
        }
    else:
        assert method_two_out is not None
        result = {
            "gold": gold, "pred": pred, "l1_ok": l1_ok, "l2_ok": l2_ok, "l3_ok": l3_ok, "usage": usage,
            "raw_output": method_two_out["raw_output"], "prompt": method_two_out["prompt"],
            "time_s": request_time_s,
        }
    if debug:
        gold_names = (id2name.get(gold[0], "?"), id2name.get(gold[1], "?"), id2name.get(gold[2], "?"))
        pred_names = (id2name.get(pred[0], "?") if pred[0] is not None else "?", id2name.get(pred[1], "?") if pred[1] is not None else "?", id2name.get(pred[2], "?") if pred[2] is not None else "?")
        print(f"[DEBUG eval] sample {i}: gold={gold} {gold_names} | pred={pred} {pred_names} | L1_ok={l1_ok} L2_ok={l2_ok} L3_ok={l3_ok}")
    return i, result


def run_eval(samples, method, id2name, hierarchy, num_levels=3, system_prompt_one=None, debug=False, concurrency=10, model=None, checkpoint_path=None):
    """对样本列表跑分类，返回 (results, elapsed)。支持断点续跑。"""
    model = model or MODEL_DEFAULT
    if not samples:
        return [], 0.0

    # 加载断点
    results_by_idx = {}
    if checkpoint_path:
        loaded, ck_method, ck_model = load_checkpoint(checkpoint_path)
        if loaded:
            results_by_idx = dict(loaded)
            if ck_method == method and ck_model == model:
                print(f"[Checkpoint] Loaded {len(results_by_idx)} completed samples from {checkpoint_path}")
            else:
                print(f"[Checkpoint] Method/model mismatch (ck: {ck_method}/{ck_model}), ignoring loaded results")
                results_by_idx = {}

    # 只对未完成的样本提交任务
    pending_indices = [i for i in range(len(samples)) if i not in results_by_idx]
    if not pending_indices:
        print("[Checkpoint] All samples already completed, skipping API calls")
        results = [results_by_idx[i] for i in range(len(samples))]
        return results, 0.0

    task_args = [
        (i, obj, method, id2name, hierarchy, num_levels, system_prompt_one, debug, model)
        for i, obj in enumerate(samples)
        if i in pending_indices
    ]
    lock = threading.Lock()
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_classify_one_sample, a): a[0] for a in task_args}
        iterator = as_completed(futures)
        if not debug:
            iterator = tqdm(iterator, total=len(futures), desc="classify", unit="sample")
        for future in iterator:
            i, result = future.result()
            with lock:
                results_by_idx[i] = result
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, results_by_idx, method, model)

    t_end = time.time()
    results = [results_by_idx[i] for i in range(len(samples))]
    elapsed = t_end - t_start
    return results, elapsed


def aggregate_stats(results, include_tokens=True):
    """n, time（单样本请求时间汇总）, acc_l1, acc_l2, acc_l3（层级一致），及 token 汇总。"""
    n = len(results)
    if n == 0:
        out = {"n": 0, "time_s_sum": 0, "time_s_avg": 0, "acc_l1": 0, "acc_l2": 0, "acc_l3": 0}
        if include_tokens:
            out["prompt_tokens"] = out["completion_tokens"] = out["total_tokens"] = 0
        return out
    l1_ok = sum(1 for r in results if r["l1_ok"])
    l2_ok = sum(1 for r in results if r["l2_ok"])
    l3_ok = sum(1 for r in results if r["l3_ok"])
    time_s_sum = sum(r.get("time_s", 0) for r in results)
    time_s_avg = time_s_sum / n if n else 0
    out = {
        "n": n,
        "time_s_sum": time_s_sum,
        "time_s_avg": time_s_avg,
        "acc_l1": l1_ok / n,
        "acc_l2": l2_ok / n,
        "acc_l3": l3_ok / n,
        "count_l1_ok": l1_ok,
        "count_l2_ok": l2_ok,
        "count_l3_ok": l3_ok,
    }
    if include_tokens:
        pt = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
        ct = sum(r.get("usage", {}).get("completion_tokens", 0) for r in results)
        tt = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)
        out["prompt_tokens"] = pt
        out["completion_tokens"] = ct
        out["total_tokens"] = tt
    return out


def main():
    global _timeout_s, _max_retries
    args = parse_args()
    _timeout_s = args.timeout
    _max_retries = args.max_retries

    id2name = load_id2name(args.labels)
    hierarchy = load_hierarchy(args.hierarchy)

    with open(args.input_sampled, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    num_levels = len(all_samples[0]["labels"]) if all_samples else 3
    if num_levels not in (2, 3):
        num_levels = 3

    # 只跑一组实验：对前 max(sample_sizes) 条做分类，再按 100/500/1000 子集分别统计准确率（非三组独立实验）
    sample_sizes = [int(x) for x in args.samples.split(",")]
    max_n = max(sample_sizes)
    samples = all_samples[:max_n]

    system_prompt_one = None
    if args.method == "one":
        system_prompt_one = get_instruct_full(id2name, hierarchy, num_levels=num_levels)

    ck_info = f", checkpoint={args.checkpoint}" if args.checkpoint else ""
    print(f"Running method={args.method}, model={args.model}, num_levels={num_levels}, max samples={len(samples)}, concurrency={args.concurrency}, timeout={args.timeout}s, max_retries={args.max_retries}{ck_info}" + (" [DEBUG]" if args.debug else ""))
    results, total_elapsed = run_eval(
        samples, args.method, id2name, hierarchy,
        num_levels=num_levels, system_prompt_one=system_prompt_one,
        debug=args.debug, concurrency=args.concurrency, model=args.model,
        checkpoint_path=args.checkpoint if args.checkpoint else None,
    )

    # 按 100, 500, 1000 子集统计（含 token）；time_s 为该子集内单样本请求时间之和
    report = []
    for size in sample_sizes:
        if size > len(results):
            continue
        sub = results[:size]
        stats = aggregate_stats(sub)
        entry = {
            "n": size,
            "time_s_sum": stats["time_s_sum"],
            "time_s_avg": stats["time_s_avg"],
            "acc_l1": stats["acc_l1"],
            "acc_l2": stats["acc_l2"],
            "acc_l3": stats["acc_l3"],
            "count_l1_ok": stats["count_l1_ok"],
            "count_l2_ok": stats["count_l2_ok"],
            "count_l3_ok": stats["count_l3_ok"],
        }
        entry["prompt_tokens"] = stats.get("prompt_tokens", 0)
        entry["completion_tokens"] = stats.get("completion_tokens", 0)
        entry["total_tokens"] = stats.get("total_tokens", 0)
        report.append(entry)

    # 总耗时对应整批；各 size 的准确率独立；写入每条样本的真值与预测 label id 便于校验
    full_stats = aggregate_stats(results)
    per_sample = []
    for i, r in enumerate(results):
        g = r["gold"]
        gold_names = [id2name.get(g[0], "?"), id2name.get(g[1], "?"), id2name.get(g[2], "?") if g[2] is not None else None]
        entry = {
            "idx": i,
            "time_s": r.get("time_s"),
            "gold": list(r["gold"]),
            "gold_names": gold_names,
            "pred": list(r["pred"]),
            "l1_ok": r["l1_ok"],
            "l2_ok": r["l2_ok"],
            "l3_ok": r["l3_ok"],
            "raw_output": r.get("raw_output"),
        }
        if args.method == "one":
            entry["user_prompt"] = r.get("user_prompt")
        else:
            entry["prompt"] = r.get("prompt")
        per_sample.append(entry)
    total_request_time_s = sum(r.get("time_s", 0) for r in results)
    out = {
        "method": args.method,
        "model": args.model,
        "wall_time_s": total_elapsed,
        "total_request_time_s": total_request_time_s,
        "total_n": len(results),
        "by_sample_size": report,
        "full": full_stats,
        "per_sample": per_sample,
    }
    if args.method == "one" and system_prompt_one:
        out["prompt"] = {"system": system_prompt_one}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Wall time (s):", total_elapsed)
    print("Total request time (sum of per-sample time_s):", total_request_time_s)
    print("By sample size:", report)
    # 汇总表格：单样本请求时间（sum/avg）、token、每级准确率
    print("\n--- 汇总表格（单样本请求时间 + 层级一致准确率 + token）---")
    print(f"{'n':>6} | {'time_sum':>10} | {'time_avg':>10} | {'total_tokens':>12} | {'acc_l1':>8} | {'acc_l2':>8} | {'acc_l3':>8}")
    print("-" * 80)
    for r in report:
        ts_sum = r.get("time_s_sum", 0)
        ts_avg = r.get("time_s_avg", 0)
        tt = r.get("total_tokens", 0)
        print(f"{r['n']:>6} | {ts_sum:>10.2f} | {ts_avg:>10.2f} | {tt:>12} | {r['acc_l1']:>8.4f} | {r['acc_l2']:>8.4f} | {r['acc_l3']:>8.4f}")
    print("Results written to", args.output)


if __name__ == "__main__":
    main()
