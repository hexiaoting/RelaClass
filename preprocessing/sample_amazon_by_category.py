"""
从 data/Amazon/test.json 按三级组合类别（labels 整条，如 39-40-44）均衡分层抽样，
输出指定数量（如 1000 或 2000）的样本，使各三级组合类别尽量均衡覆盖。
"""
import argparse
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stratified sample from Amazon test.json by full 3-level label (tuple of labels)"
    )
    parser.add_argument("--input", type=str, default="data/Amazon/test.json",
                        help="Path to test.json (JSONL)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path, e.g. data/Amazon/test_sampled_1000.json")
    parser.add_argument("--n", type=int, default=1000, choices=[1000, 2000],
                        help="Target number of samples (1000 or 2000)")
    return parser.parse_args()


def _print_count_stats(counts, prefix=""):
    """打印每类数量的 min/max/均值 以及 样本数=1,2,... 的类别数分布。"""
    if not counts:
        print(f"{prefix} 无数据")
        return
    vals = list(counts)
    n = len(vals)
    print(f"{prefix} 每类样本数: min={min(vals)}, max={max(vals)}, mean={sum(vals)/n:.2f}")
    dist = defaultdict(int)
    for v in vals:
        dist[v] += 1
    print(f"{prefix} 分布 (样本数 -> 类别数): {dict(sorted(dist.items()))}")


def main():
    args = parse_args()
    if args.output is None:
        args.output = f"data/Amazon/test_sampled_{args.n}.json"

    # 按三级组合 (l1, l2, l3) 分组
    by_category = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cat_key = tuple(obj["labels"])
            by_category[cat_key].append(obj)

    n_categories = len(by_category)
    target = args.n
    # 每类目标数量（向下取整）；若某类不足则取尽，余量再轮流补足
    per_class = target // n_categories
    sampled = []
    used_per_category = {cat: set() for cat in by_category}

    def select_from_class(cat_key, k):
        """从类别 cat_key 中均匀选 k 个，返回选中的样本及索引"""
        items = by_category[cat_key]
        if k <= 0:
            return [], []
        if k >= len(items):
            return items[:], list(range(len(items)))
        step = len(items) / k
        indices = [int(i * step) for i in range(k)]
        return [items[i] for i in indices], indices

    for cat_key, items in by_category.items():
        take = min(len(items), per_class)
        chosen, indices = select_from_class(cat_key, take)
        sampled.extend(chosen)
        used_per_category[cat_key] = set(indices)

    # 若总数不足 target，从各类剩余样本中轮流补足
    total_taken = len(sampled)
    if total_taken < target:
        shortfall = target - total_taken
        class_keys = list(by_category.keys())
        idx = 0
        while shortfall > 0:
            c = class_keys[idx % len(class_keys)]
            available = [i for i in range(len(by_category[c])) if i not in used_per_category[c]]
            if available:
                i = available[0]
                used_per_category[c].add(i)
                sampled.append(by_category[c][i])
                shortfall -= 1
            idx += 1
            if idx > len(class_keys) * max(len(by_category[ci]) for ci in class_keys):
                break

    sampled = sampled[:target]

    with open(args.output, "w", encoding="utf-8") as f:
        for obj in sampled:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ----- 原始数据统计 -----
    print("========== 原始数据统计 ==========")
    print(f"原始数据类别总数（三级组合数）: {n_categories}")
    original_counts = [len(items) for items in by_category.values()]
    _print_count_stats(original_counts, "原始数据")

    # ----- 采样结果统计 -----
    print("========== 采样结果统计 ==========")
    sampled_cat_counts = defaultdict(int)
    for obj in sampled:
        sampled_cat_counts[tuple(obj["labels"])] += 1
    n_sampled_cats = len(sampled_cat_counts)
    print(f"采样结果类别总数: {n_sampled_cats}")
    print(f"采样总样本数: {len(sampled)} -> {args.output}")
    sampled_counts = list(sampled_cat_counts.values())
    _print_count_stats(sampled_counts, "采样结果")


if __name__ == "__main__":
    main()
