# import numpy as np
# import torch
# import torch.utils.data as data
import os
import gc
import torch
import datasets
import numpy
# from datasets import load_from_disk

import pickle
import torch.nn.functional as F
from torch import Tensor
import argparse
from instruct import amazon_531_instruct_task123,amazon_531_instruct_task12
from instruct import wos_task_en, dbpedia_task_en,normal_task
from instruct import amazon_531_food_instruct_task,amazon_531_toysgames_instruct_task,amazon_531_beauty_instruct_task,amazon_531_healthpersonalcare_instruct_task,amazon_531_babyproducts_instruct_task,amazon_531_petsupplies_instruct_task,amazon_531_beauty_instruct_task_v2,amazon_531_beauty_hair_instruct_task, amazon_531_beauty_hair_instruct_task_v3
from instruct import hwt_dataset_all_instruct_task1, hwt_dataset_all_instruct_task12
from instruct import wos5736_instruct_task12, wos5736_instruct_task2, wos_instruct_task12, wos_instruct_task1, wos11967_instruct_task12, wos11967_instruct_task12_v2
from instruct import amazon_531_food_simple_instruct_task, amazon_531_food_instruct_task_12, amazon_531_petsupplies_instruct_task1
from instruct import amazon_531_petsupplies_cat_instruct_task2
from instruct import dbpedia_task0, dbpedia_task12, dbpedia_task1
from instruct import hwt_dataset_265_instruct_task12

type_2_task={
    'amazon531-0':amazon_531_food_instruct_task,
    'amazon531-0-task1':amazon_531_food_simple_instruct_task,
    'amazon531-0-task12':amazon_531_food_instruct_task_12,
    'amazon531-3':amazon_531_toysgames_instruct_task,
    'amazon531-10':amazon_531_beauty_instruct_task,
    'amazon531-10-64':amazon_531_beauty_hair_instruct_task,
    'amazon531-10-64-v3':amazon_531_beauty_hair_instruct_task_v3,
    'amazon531-10-v2':amazon_531_beauty_instruct_task_v2,
    'amazon531-23':amazon_531_healthpersonalcare_instruct_task,
    'amazon531-40':amazon_531_babyproducts_instruct_task,
    'amazon531-169':amazon_531_petsupplies_instruct_task    ,
    'amazon531-169-task1': amazon_531_petsupplies_instruct_task1,
    'amazon531-169-cat-task2': amazon_531_petsupplies_cat_instruct_task2,
    'amazon531-task-123':amazon_531_instruct_task123,
    'amazon531-task-12':amazon_531_instruct_task12,
    'wos5736-task-12':wos5736_instruct_task12,
    'wos5736-task-2':wos5736_instruct_task2,
    'wos-task-12':wos_instruct_task12,
    'wos-task-1':wos_instruct_task1,
    'wos11967-task-12':wos11967_instruct_task12,
    'wos11967-task-12-v2':wos11967_instruct_task12_v2,
    'hwtdataset-all-task-1': hwt_dataset_all_instruct_task1,
    'hwtdataset-all-task-12': hwt_dataset_all_instruct_task12,
    'hwtdataset-amazon-265-task-12': hwt_dataset_265_instruct_task12,
    'task-0':normal_task,
    'dbpedia-task0':dbpedia_task0,
    'dbpedia-task1':dbpedia_task1,
    'dbpedia-task12':dbpedia_task12,
}
#Amazon子分类个数分别为16, 17, 6, 7, 12, 6
task = None
def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, default="SFR-Embedding-2_R",help='"bge-large-en-v1.5 / bge-en-icl / SFR-Embedding-2_R')
    parser.add_argument('--task_type' , type=str, required=True, help='For Amazon531: full/0/3/10/23/40/169')
    parser.add_argument('--specific_file', type=str, required=False)
    parser.add_argument('--columns', type=str, default="token", help="token;extracted_keywords")
    parser.add_argument('--dataset_save_dir', type=str, help='dataset saving dir, like /home/hewenting/data_preprocess/Amazon-531/SFR-Embedding-2_R_28932_merged')

    return parser.parse_args()

def get_task(dataset_name, task_type):
    if dataset_name == "Amazon-531":
        if task_type in type_2_task.keys():
            task = type_2_task[task_type]
        else:
            raise Exception(f"Not support this task_type {task_type}")
    elif dataset_name == "WebOfScience":
        task = type_2_task[task_type]
    elif dataset_name == "DBPedia-298":
        task = type_2_task[task_type]
        #task = dbpedia_task_en
    elif dataset_name == "hwt_dataset":
        task = type_2_task[task_type]
    else:
        raise Exception("Task not set")
    
    return task

#按照模型的指令要求，生成相应的数据格式。
def get_detailed_instruct(task_description: str, queries, model_name="SFR-Embedding-2_R"):
    results = []
    for q in queries:
        if model_name == "SFR-Embedding-2_R":
            results.append(f'Instruct: {task_description}\nQuery: {q}')
        elif model_name == "bge-en-icl":
            results.append(f'<instruct>{task_description}\n<query>{q}')
    return results

def get_new_queries(queries, query_max_len, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
            tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False
    )
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs['input_ids'])
    for i in range(len(new_queries)):
        new_queries[i] = new_queries[i] + '\n<response>'
    return new_max_length, new_queries

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def pad_sequences(seq, target_len=3, pad_value=-1):
    return seq + [pad_value] * (target_len - len(seq))

#将每个batch的训练样本的columns字段内容，通过model进行embedding
#注意columns是一个数组，可以包含多个列名，那就会分别针对这些列名进行embedding，保存下来
# 如 columns和embedding字段的关系(以SFR-Embedding-2_R模型为例)：
#       original_token  --->    embedding_original_token, embedding_original_token_normalize
#       llm_keywords_analysis   --->    embedding_llm_keywords_analysis, embedding_llm_keywords_analysis_normalize
#       token           --->    embedding,embedding_normalize 【columns默认为token】
def data_map_embedding_function(dataset_name, batch, tokenizer, model, model_name, columns=['token']):
    #不同的数据集采用的指令不一样，需要定制化
    new_batch = {}
    new_batch['labels'] = [pad_sequences(l) for l in batch['label']]

    for column_name in columns:
        embedding_name = 'embedding' if column_name =='token' else f"embedding_{column_name}"
        if model_name == "bert-base-uncased":
            tokens = tokenizer(batch[column_name], padding=True, truncation=True, max_length = 512,return_tensors='pt')
            tokens = {key: value.to("cuda:0") for key, value in tokens.items()}
            new_batch[embedding_name] = model(**tokens).last_hidden_state[:, 0].cpu().detach()
        elif model_name == "bge-large-en-v1.5":
            tokens = tokenizer(batch[column_name], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
            with torch.no_grad():
                new_batch[embedding_name] = torch.nn.functional.normalize(model(**tokens)[0][:, 0].cpu().detach(), p=2, dim=1)
        elif model_name == "SFR-Embedding-2_R":
            #重新写instruct的版本
            input_texts = get_detailed_instruct(task, batch[column_name], model_name="SFR-Embedding-2_R")
            tokens = tokenizer(input_texts, max_length=4096, padding=True, truncation=True, return_tensors="pt").to("cuda:0")
            outputs = model(**tokens)
            new_batch[embedding_name] = last_token_pool(outputs.last_hidden_state, tokens['attention_mask'])
            # normalize embeddings
            new_batch[f'{embedding_name}_normalize'] = F.normalize(new_batch[embedding_name], p=2, dim=1)
        elif model_name == "bge-en-icl":
            input_texts = get_detailed_instruct(task, batch[column_name], model_name="bge-en-icl")
            query_max_len = 512
            new_query_max_len, new_queries = get_new_queries(input_texts, query_max_len, tokenizer)

            tokens = tokenizer(new_queries, max_length=new_query_max_len, padding=True, truncation=True,return_tensors='pt').to("cuda:0")

            with torch.no_grad():
                query_outputs = model(**tokens)
                new_batch[embedding_name] = last_token_pool(query_outputs.last_hidden_state, tokens['attention_mask'])

            # normalize embeddings
            new_batch[f'{embedding_name}_normalize'] = F.normalize(new_batch[embedding_name], p=2, dim=1)


    del tokens
    torch.cuda.empty_cache()
    gc.collect()
    return new_batch

def generate_mapping(labels, process_3rd_mapping=True):
    # import ipdb
    # ipdb.set_trace()
    mapping = {}
    if isinstance(labels, numpy.ndarray) or isinstance(labels, torch.Tensor):
        labels= labels.tolist()

    l1_values = sorted(set([label[0] for label in labels]))

    if max(l1_values) != len(l1_values) - 1: #说明第一级分类标签id不是从0开始的顺序编号，需要做映射
        tmp = {}
        for idx, v in enumerate(l1_values):
            tmp[v] = idx
        mapping['level-0-mapping'] = tmp
    elif l1_values==[0]:
        mapping['level-0-mapping'] = {0: 0}
    else:
        tmp = {}
        for idx, v in enumerate(l1_values):
            tmp[v] = idx
        mapping['level-0-mapping'] = tmp

    cur_idx = 0
    offset_l3 = 0
    for i in l1_values:
        #获得第一级分类标签为i，的第二类标签值
        l2_values = [label[1] for label in labels if label[0] == i]
        l2_sorted_values = sorted(set(l2_values))

        l2_cb_id = {}
        for idx, v in enumerate(l2_sorted_values):
            l2_cb_id[v] = cur_idx + idx
            
            #处理第三级分类信息：
            if not process_3rd_mapping:
                continue
            l3_cb_id = {}

            l3_values = [label[2] for label in labels if label[1] == v]
            l3_sorted_values = sorted(set(l3_values))
            if l3_sorted_values[0] == -1:
                l3_sorted_values = l3_sorted_values[1:]
            for idx_l3, v3 in enumerate(l3_sorted_values):
                l3_cb_id[v3] = idx_l3 + offset_l3
            offset_l3 += len(l3_sorted_values)
            mapping[v] = l3_cb_id
            #处理完毕
            
        cur_idx += idx + 1
        mapping[i] = l2_cb_id
    return mapping
    # print(mapping)
    #WebOfScience数据集（7大类别）举例：{0: {7: 0, 12: 1, 23: 2, 28: 3, 37: 4, 41: 5, 46: 6, 49: 7, 50: 8, 57: 9, 91: 10, 99: 11, 101: 12, 110: 13, 111: 14, 125: 15, 140: 16}, 1: {8: 0, 11: 1, 16: 2, 19: 3, 21: 4, 22: 5, 27: 6, 35: 7, 36: 8, 38: 9, 39: 10, 40: 11, 43: 12, 44: 13, 47: 14, 52: 15, 53: 16, 54: 17, 63: 18, 65: 19, 68: 20, 69: 21, 70: 22, 72: 23, 73: 24, 74: 25, 75: 26, 76: 27, 78: 28, 80: 29, 85: 30, 89: 31, 106: 32, 107: 33, 108: 34, 109: 35, 112: 36, 114: 37, 115: 38, 116: 39, 117: 40, 121: 41, 124: 42, 126: 43, 127: 44, 128: 45, 129: 46, 131: 47, 132: 48, 134: 49, 135: 50, 138: 51, 139: 52}, 2: {9: 0, 33: 1, 45: 2, 51: 3, 56: 4, 66: 5, 71: 6, 82: 7, 83: 8, 118: 9, 120: 10}, 3: {10: 0, 24: 1, 29: 2, 55: 3, 59: 4, 60: 5, 62: 6, 79: 7, 84: 8, 90: 9, 95: 10, 98: 11, 102: 12, 103: 13, 113: 14}, 4: {13: 0, 17: 1, 18: 2, 20: 3, 25: 4, 26: 5, 31: 6, 32: 7, 42: 8}, 5: {14: 0, 34: 1, 48: 2, 61: 3, 67: 4, 77: 5, 81: 6, 92: 7, 94: 8}, 6: {15: 0, 30: 1, 64: 2, 86: 3, 87: 4, 88: 5, 93: 6, 96: 7, 97: 8, 100: 9, 104: 10, 105: 11, 119: 12, 122: 13, 123: 14, 130: 15, 133: 16, 136: 17, 137: 18}}
    # with open('{}/labels_mapping.pkl'.format(data_dir), 'wb') as f:
    #     pickle.dump(mapping, f)

model_path={
    'bert-base-uncased':  "/mnt/disk5/hewenting_nfs_serverdir/models/google-bert:bert-base-uncased",
    "bge-large-en-v1.5": "/mnt/disk5/hewenting_nfs_serverdir/models/baai:bge-large-en-v1.5",
    "SFR-Embedding-2_R": "/mnt/disk5/hewenting_nfs_serverdir/models/Salesforce:SFR-Embedding-2_R",
    "bge-en-icl": "/mnt/disk5/hewenting_nfs_serverdir/models/baai:bge-en-icl"
}

def data_process(dataset_name=None, 
                 model_name="SFR-Embedding-2_R", 
                 data_dir=None, 
                 with_category=False, 
                 specific_file=None, 
                 columns=["token"], 
                 column_name = "embedding_normalize",
                 dataset_save_dir=None):
    print("Using task: ",  task)
    
    train_filename = None
    if dataset_save_dir is not None:
        save_dir = dataset_save_dir
    else:
        if data_dir is None:
            data_dir = os.path.join("/home/hewenting/data_preprocess/", dataset_name)
        save_dir = os.path.join(data_dir, model_name)
        train_filename = '{}/{}_train.json'.format(data_dir, dataset_name)

    has_test_data = False
    has_dev_data = False

    if dataset_name == "Amazon-531": #这个数据集只有train和dev
        has_test_data=False
    elif dataset_name == "Amazon-392-hwt": #这个数据集只有train, 且原始数据有带分类文本的也有不带分类文本的
        has_test_data=False
        has_dev_data=False
        if with_category:
            save_dir = os.path.join(data_dir, f'{model_name}_withcategory')
            train_filename = '{}/{}_train_withcategory.json'.format(data_dir, dataset_name)
        else:
            save_dir = os.path.join(data_dir, f'{model_name}_withoutcategory')
            train_filename = '{}/{}_train_withoutcategory.json'.format(data_dir, dataset_name)
    elif dataset_name == "DBPedia-298":
        print("Using default settings")
        
    if specific_file:
        train_filename = specific_file
        if dataset_save_dir is None:
            save_dir = os.path.join(data_dir, f"{model_name}_{specific_file.split('/')[-1].split('.')[0]}")


    if os.path.exists(save_dir):
        print(f"\n----->data_process load_from_disk {save_dir}")
        dataset = datasets.load_from_disk(save_dir)
        print("columns:", dataset['train'].features.keys(), end='\n\n')
        labels = dataset['train']['labels']
        if dataset_name == 'wos':
            trimmed_labels = [sublist[:-1] for sublist in labels]
            labels = trimmed_labels
        labels_mapping = generate_mapping(labels, dataset_name != 'wos')
        # print(type(dataset['train']['embedding']), len(dataset['train']['embedding'][0]))
    else:
        print(f"\n----->data_process load_from_text & embedding, save to {save_dir}")
        print("train_filename=",train_filename)

        data_files={'train': train_filename}
        if has_dev_data:
            data_files['dev'] = '{}/{}_dev.json'.format(data_dir, dataset_name)
        if has_test_data:
            data_files['test'] = '{}/{}_test.json'.format(data_dir, dataset_name)
        dataset = datasets.load_dataset('json',  data_files=data_files)
        if not set(columns).issubset(dataset['train'].features):
            raise Exception(f"{columns} not valid, datasets.columns={dataset['train'].features}")
        print(f"Processing columns: {columns}")

        if model_name not in model_path.keys():
            raise Exception("Do not support this model")
        from transformers import AutoTokenizer, AutoModel
        #import ipdb
        #ipdb.set_trace()
        tokenizer = AutoTokenizer.from_pretrained(model_path[model_name], use_fast=False)
        model = AutoModel.from_pretrained(model_path[model_name]).cuda()
        model.eval()

        dataset = dataset.map(lambda x: data_map_embedding_function(dataset_name, x, tokenizer, model,  model_name, columns=columns), batched=True, batch_size=1)
        dataset.save_to_disk(save_dir)
        # if specific_file == None:
        labels_mapping = generate_mapping(data_dir, dataset['train']['labels'])
    
    # labels_mapping = {}
    # with open('{}/labels_mapping.pkl'.format(data_dir), 'rb') as f:
    #     labels_mapping = pickle.load(f)
    print(labels_mapping)

    delete_columns = []
    for key in dataset['train'].features.keys():
        if key not in ['labels' ,'asin','id', 'extracted_keywords', 'token',column_name]:
            delete_columns.append(key)

    dataset = dataset.remove_columns(delete_columns)

    if column_name in dataset['train'].column_names:
         dataset['train'].set_format('torch', columns=['labels', column_name], output_all_columns=True)

    if has_dev_data:
        dataset['dev'].set_format('torch', columns=['embedding', 'labels'], output_all_columns=True)
    if has_test_data:
        dataset['test'].set_format('torch', columns=['embedding', 'labels'], output_all_columns=True)

    label_dict = {}
    if data_dir and os.path.exists(os.path.join(data_dir, 'value_dict.pt')):
        label_dict = torch.load(os.path.join(data_dir, 'value_dict.pt'))   #data/WebOfScience/value_dict.pt
        label_dict = {i: v for i, v in label_dict.items()}
        # webofscience数据集的label_dict共141个key,其中0-6是父类,label_dict={0: 'CS', 1: 'Medical', 2: 'Civil', 3: 'ECE', 4: 'biochemistry', 5: 'MAE', 6: 'Psychology', 7: 'Symbolic computation', 8: "Alzheimer's Disease", 9: 'Green Building', 10: 'Electric motor', 11: "Parkinson's Disease", 12: 'Computer vision', 13: 'Molecular biology', 14: 'Fluid mechanics', 15: 'Prenatal development', 16: 'Sprains and Strains', 17: 'Enzymology', 18: 'Southern blotting', 19: 'Cancer', 20: 'Northern blotting', 21: 'Sports Injuries', 22: 'Senior Health', 23: 'Computer graphics', 24: 'Digital control', 25: 'Human Metabolism', 26: 'Polymerase chain reaction', 27: 'Multiple Sclerosis', 28: 'Operating systems', 29: 'Microcontroller', 30: 'Attention', 31: 'Immunology', 32: 'Genetics', 33: 'Water Pollution', 34: 'Hydraulics', 35: 'Hepatitis C', 36: 'Weight Loss', 37: 'Machine learning', 38: 'Low Testosterone', 39: 'Fungal Infection', 40: 'Diabetes', 41: 'Data structures', 42: 'Cell biology', 43: 'Parenting', 44: 'Birth Control', 45: 'Smart Material', 46: 'network security', 47: 'Heart Disease', 48: 'computer-aided design', 49: 'Image processing', 50: 'Parallel computing', 51: 'Ambient Intelligence', 52: 'Allergies', 53: 'Menopause', 54: 'Emergency Contraception', 55: 'Electrical network', 56: 'Construction Management', 57: 'Distributed computing', 58: 'Electrical generator', 59: 'Electricity', 60: 'Operational amplifier', 61: 'Manufacturing engineering', 62: 'Analog signal processing', 63: 'Skin Care', 64: 'Eating disorders', 65: 'Myelofibrosis', 66: 'Suspension Bridge', 67: 'Machine design', 68: 'Hypothyroidism', 69: 'Headache', 70: 'Overactive Bladder', 71: 'Geotextile', 72: 'Irritable Bowel Syndrome', 73: 'Polycythemia Vera', 74: 'Atrial Fibrillation', 75: 'Smoking Cessation', 76: 'Lymphoma', 77: 'Thermodynamics', 78: 'Asthma', 79: 'State space representation', 80: 'Bipolar Disorder', 81: 'Materials Engineering', 82: 'Stealth Technology', 83: 'Solar Energy', 84: 'Signal-flow graph', 85: "Crohn's Disease", 86: 'Borderline personality disorder', 87: 'Prosocial behavior', 88: 'False memories', 89: 'Idiopathic Pulmonary Fibrosis', 90: 'Electrical circuits', 91: 'Algorithm design', 92: 'Strength of materials', 93: 'Problem-solving', 94: 'Internal combustion engine', 95: 'Lorentz force law', 96: 'Prejudice', 97: 'Antisocial personality disorder', 98: 'System identification', 99: 'Computer programming', 100: 'Nonverbal communication', 101: 'Relational databases', 102: 'PID controller', 103: 'Voltage law', 104: 'Leadership', 105: 'Child abuse', 106: 'Mental Health', 107: 'Dementia', 108: 'Rheumatoid Arthritis', 109: 'Osteoporosis', 110: 'Software engineering', 111: 'Bioinformatics', 112: 'Medicare', 113: 'Control engineering', 114: 'Psoriatic Arthritis', 115: 'Addiction', 116: 'Atopic Dermatitis', 117: 'Digestive Health', 118: 'Remote Sensing', 119: 'Gender roles', 120: 'Rainwater Harvesting', 121: 'Healthy Sleep', 122: 'Depression', 123: 'Social cognition', 124: 'Anxiety', 125: 'Cryptography', 126: 'Psoriasis', 127: 'Ankylosing Spondylitis', 128: "Children's Health", 129: 'Stress Management', 130: 'Seasonal affective disorder', 131: 'HIV/AIDS', 132: 'Migraine', 133: 'Person perception', 134: 'Osteoarthritis', 135: 'Hereditary Angioedema', 136: 'Media violence', 137: 'Schizophrenia', 138: 'Kidney Health', 139: 'Autism', 140: 'Structured Storage'}
    elif data_dir and os.path.exists(os.path.join(data_dir, 'train/labels.txt')):
        with open(os.path.join(data_dir, 'train/labels.txt'), 'r', encoding='utf-8') as file:
            for line in file:
                id, label = line.strip().split('\t')
                label_dict[int(id)] = label

    return dataset, labels_mapping, label_dict


if __name__ == '__main__':
    args = parse_args()
    print(args)
    task = get_task(args.dataset_name, args.task_type)
    

    # dataset_name = "WebOfScience"
    # data_dir = "/mnt/disk5/hewenting_nfs_serverdir/githubs/HPT/data/WebOfScience/results_with_keywords"

    dataset, labels_mapping, label_dict = data_process(
                            args.dataset_name,
                            model_name=args.embedding_model,
                            data_dir = None,
                            specific_file=args.specific_file,
                            columns = args.columns.split(";"),
                            dataset_save_dir = args.dataset_save_dir)
