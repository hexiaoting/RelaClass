import os
from argparse import ArgumentParser
import json
from tqdm import tqdm
import pickle

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=False, default="/mnt/disk5/hewenting_nfs_serverdir/datasets/amazon-review/raw_data/metadata_2014")
    parser.add_argument('--datasets', type=str, required=False, default='Beauty;Musical_Instruments;Sports_and_Outdoors;Toys_and_Games')
    parser.add_argument('--output_dir', type=str, required=False, default="/home/hewenting/data_preprocess/Amazon-4type-hwt")
    parser.add_argument('--max_rows', type=int, required=False, default='10000')
    parser.add_argument('--output_filename', type=str, default=None)
    parser.add_argument('--use_category_info', type=int, default=0, help="0 means not , 1 means yes")
    args = parser.parse_args()
    return args


# 返回的new_text说明：
#   商品必须得有description信息，否则就为不要这个商品
#   如何args里设置了use_category_info, 就必须得有分类字段，如果没有，那么这个item信息也不要
def text_process(use_category_info,dataset_name, item, text_seq="tdc", save_category=True):
    import ipdb
    # ipdb.set_trace()
    if 'description' not in item.keys() or 'title' not in item.keys():
        return "",""

    new_text = []

    # title = f'title: {item["title"]}'
    # description = f'description: {item["description"]}'
    new_text.append(item["title"])
    if isinstance(item["description"], str):
        new_text.append(item["description"])
    elif isinstance(item["description"], list):
        if len(item["description"]) != 1:
            return "",""
        new_text.append(item["description"][0])


    if dataset_name not in ["instruments" , "Instruments", "Musical_Instruments"]:
        key = "categories"
        if key not in item.keys() or len(item[key]) != 1:
            # print(f"This item has no category or belongs to multiple category, skip it. item={item}")
            return "",""
        category = f'category: {", ".join(item["categories"][0])}'
        category_info = item["categories"][0]
    else:
        key = "category"
        if key not in item.keys() or len(item[key]) == 0:
            # print(f"No {key} in item:", item)
            return "",""
        if item[key][0] != "Musical Instruments":
            raise Exception(f"instruments category error:{item[key]}", e)
        category =f'category: {", ".join(item["category"])}'
        category_info = item["category"]

    if use_category_info == 1:
        new_text.append(category)

    return ', '.join(new_text), category_info


category_mapping={}
level_max=[0,0,0]

#默认只处理三级的分类
#将文本分类信息转为编号，每一级分类都是从0开始的一串连续的数字
def process_category_from_text2int(category_info):
    label = []
    for i, name in enumerate(category_info):
        if i >= 3:
            break
        if name not in category_mapping.keys():
            category_mapping[name] = level_max[i]
            level_max[i] += 1

        label.append(category_mapping[name])
    if len(label) < 3:
        label.append(-1)

    return label

def process_text_and_labels(args):
    dataset_names = args.datasets.split(';')
    datasets = {}

    for dataset_name in dataset_names:
        datasets[dataset_name] = []
        count = 0
        meta_file = os.path.join(args.root_dir, f'meta_{dataset_name}.json')
        with open(meta_file) as f:
            readin = f.readlines()
            for line in tqdm(readin):
                item = eval(line)

                new_text, category_info = text_process(args.use_category_info, dataset_name, item)
                if new_text != "" and len(category_info) >= 3:
                    label = process_category_from_text2int(category_info)
                    datasets[dataset_name].append({"text": new_text,"label":label})
                    count += 1
                    if count > args.max_rows:
                        break


    results = []
    for i in range(args.max_rows):
        for dataset_name in dataset_names:
            item={}
            item['token'] = datasets[dataset_name][i]['text']
            item['label'] = datasets[dataset_name][i]['label']
            results.append(item)

    if args.output_filename == None:
        output_file = os.path.join(args.output_dir, f'Amazon-392-hwt_train.json')
    else:
        output_file = os.path.join(args.output_dir, args.output_filename)

    with open(output_file, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    args = parse_args()
    process_text_and_labels(args)
    print(category_mapping)
    with open('{}/labels_mapping.pkl'.format(args.output_dir), 'wb') as f:
        pickle.dump(category_mapping, f)
