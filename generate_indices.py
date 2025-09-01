# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
import argparse
import numpy as np

from torch.utils.data import DataLoader
import json
import os

from models.rqvae import RQVAE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--dataset_save_dir', type=str)
    parser.add_argument('--column_name', type=str, default='embedding_normalize')
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--normal', action='store_true', help="This is a normal rqvae model")
    parser.add_argument('--cb_id', type=int)
    parser.add_argument('--save_negative_samples', action='store_true', help="如果设置，则可以将错误的样本打印出来，但是需要改trainer.py里的negative_indices")
    return parser.parse_args()

codebook_shared_status=[False, False, True]
def convert_true_lable_to_cbindex(num_emb_list, indices, labels, level=0, parent_cb_id = -1, cb_id = -1):
    # import ipdb
    # ipdb.set_trace()
    class_num = 0 #总分类个数
    #首先获得labels当前这一级别(level级别)的不同编号，映射到0,1,2,...
    lableid2idx_mapping={}
    label = labels[:, level]
    if isinstance(label, np.ndarray):
        sorted_label = np.unique(label)
        if len(sorted_label) == 0:
                return {'new_label':None, 'correct_sample_num':[0,0,0], 'codebook_order':['','',''], 'distinct_class_num':[0,0,0]}
        if  sorted_label[0] == -1:
            sorted_label = sorted_label[1:]
        class_num = len(sorted_label)
        for idx, v in enumerate(sorted_label):
            lableid2idx_mapping[idx] = v
    else:
        label = label.cpu()
        sorted_tensor, idx = torch.unique(label).sort()
        class_num = idx.numpy()[-1]+1
        for i in idx.numpy():
            lableid2idx_mapping[i] = sorted_tensor[i].item()

    #lableid2idx_mapping = {11: {12: 0, 45: 1, 67: 2, 69: 3, 109: 4, 205: 5, 419: 6}, 44: {45: 6, 67: 7, 69: 8, 72: 9, 73: 10, 173: 11, 468: 12}, 54: {55: 12, 83: 13, 95: 14, 173: 15, 322: 16}, 60: {61: 18, 92: 19, 197: 20, 234: 21, 239: 22, 414: 23, 489: 24}, 64: {65: 24, 79: 25, 80: 26, 90: 27, 93: 28, 335: 29, 338: 30, 442: 31, 448: 32, 461: 33, 473: 34}, 220: {221: 30, 242: 31, 510: 32}, 10: {11: 0, 44: 1, 54: 2, 60: 3, 64: 4, 220: 5}}
    #if level == 1:
    #    lableid2idx_mapping= {1:65,2:79,2:80,3:90,4:93,5:335,6:338,7:442,9:448,10:461,11:473}
    

    #S2. 分别打印每一个类别通过码本获得的索引编号，比如看看beauty这个一类别在码本中分别被映射到哪个码本，每个码本映射的数量是多少
    #    最佳的结果是所有beauty的样本的码本索引都一样，toys类别的样本的码本indices都一样（但和beauty是不一样的），这才是最佳结果
    new_label = np.zeros(label.shape[0], dtype=np.int64)
    if level == 0:
        print("codebook-l0-result:")

    distinct_class_num_2nd = 0
    distinct_class_num_3rd = 0
    correct_samples_num_1st = {}
    correct_samples_num_2nd = 0
    correct_samples_num_3rd = 0
    codebook_order_1st = ""
    codebook_order_2nd = ""
    codebook_order_3rd = ""
    for i in range(class_num):
        if level == 2 and codebook_shared_status[2]:
            break
        index = np.where(label == lableid2idx_mapping[i])[0]
        unique_elements, counts = np.unique(indices[:,level][index], return_counts=True)
        sorted_counts = sorted(zip(unique_elements, counts), key=lambda x: x[1], reverse=True)
        # 用以下方法可以打印某些二级分类情况
        print("\t" * level, sorted_counts[:6])
        # if level == 0 and (i == 2 or i == 3):
        start_tmp = 0
        if level == 1:
            start_tmp = sum(num_emb_list[1: 1+ parent_cb_id])
        elif level == 2:
            cb3rd_start = num_emb_list[0] + 1
            start_tmp = sum(num_emb_list[cb3rd_start: cb3rd_start+ parent_cb_id])

        sub_index = index[np.where(indices[:,level][index] == (start_tmp + i))]
        if level == 0:
            if cb_id is not None and cb_id != -1:
                sub_index = index[np.where(indices[:,level][index] == cb_id)]
             #在这个一级码本 i 下正确的分类的样本下标，看看他们的二级分类效果如何
            codebook_2nd_res = convert_true_lable_to_cbindex(num_emb_list, indices[sub_index], 
                                                                      labels[sub_index], 
                                                                      level+1, 
                                                                      parent_cb_id = start_tmp + i)
            codebook_order_2nd += (codebook_2nd_res['codebook_order'][0] + ",")
            codebook_order_3rd += (codebook_2nd_res['codebook_order'][1] + ",")
            correct_samples_num_2nd += codebook_2nd_res['correct_sample_num'][0]
            correct_samples_num_3rd += codebook_2nd_res['correct_sample_num'][1]
            distinct_class_num_2nd += codebook_2nd_res['distinct_class_num'][0]
            distinct_class_num_3rd += codebook_2nd_res['distinct_class_num'][1]
        elif level == 1 and labels.shape[1] > 2:
            codebook_3rd_res = convert_true_lable_to_cbindex(num_emb_list, indices[sub_index], 
                                                                      labels[sub_index], 
                                                                      level+1, 
                                                                      parent_cb_id = start_tmp + i)
            codebook_order_2nd += (codebook_3rd_res['codebook_order'][0] + ",")
            correct_samples_num_2nd += codebook_3rd_res['correct_sample_num'][0]
            distinct_class_num_2nd += codebook_3rd_res['distinct_class_num'][0]
        if level == 0:
            codebook_order_1st += str(sorted_counts[0][0])
        elif level >= 1:
            codebook_order_1st += (str(sorted_counts[0][0]) + '-')

        for aa in sorted_counts:
            if aa[0] == start_tmp + i:
                correct_samples_num_1st[aa[0]] = aa[1]
        
    res = {}
    res['new_label'] = new_label
    res['distinct_class_num'] = [len(correct_samples_num_1st.keys()), distinct_class_num_2nd, distinct_class_num_3rd]
    res['correct_sample_num'] = [sum(correct_samples_num_1st.values()), correct_samples_num_2nd, correct_samples_num_3rd]
    res['codebook_order'] = [codebook_order_1st, codebook_order_2nd, codebook_order_3rd]
    return res

def aaconvert_true_lable_to_cbindex(indices, labels, level=0):
        #首先获得labels当前这一级别(level级别)的不同编号，映射到0,1,2,...
        mapping={}
        label = labels[:, level]
        if isinstance(label, np.ndarray):
            sorted_label = np.unique(label)
            size = len(sorted_label)
            for idx, v in enumerate(sorted_label):
                mapping[idx] = v
        else:
            label = label.cpu()
            sorted_tensor, idx = torch.unique(label).sort()
            size = idx.numpy()[-1]+1
            for i in idx.numpy():
                mapping[i] = sorted_tensor[i].item()

        #S2. 分别打印每一个类别通过码本获得的索引编号，比如看看beauty这个一类别在码本中分别被映射到哪个码本，每个码本映射的数量是多少
        #    最佳的结果是所有beauty的样本的码本索引都一样，toys类别的样本的码本indices都一样（但和beauty是不一样的），这才是最佳结果
        corrent_sample_num = 0
        new_label = np.zeros(label.shape[0], dtype=np.int64)
        used_codes = set()
        if level == 0:
            print("codebook-l0-result:")
        for i in range(size):
            index = np.where(label == mapping[i])[0]
            unique_elements, counts = np.unique(indices[:,level][index], return_counts=True)
            sorted_counts = sorted(zip(unique_elements, counts), key=lambda x: x[1], reverse=True)
            print(sorted_counts)
            # 用以下方法可以打印某些二级分类情况
            # print("" if level == 0 else "\t", sorted_counts)
            # if level == 0 and (i == 2 or i == 3):
            #     sub_index = index[np.where(indices[:,level][index] == i)] #在这个一级码本 i 下正确的分类的样本下标，看看他们的二级分类效果如何
            #     self.convert_true_lable_to_cbindex(indices[sub_index], labels[sub_index], level+1)
            if sorted_counts[0][0] not in used_codes:
                used_codes.add(sorted_counts[0][0])
                corrent_sample_num += sorted_counts[0][1]
            else:
                #发现两类应该区别开来的类别居然都大部分映射到同一个码去了
                print("****************Error************")
                corrent_sample_num = 0
            new_label[index]=sorted_counts[0][0]
        return new_label, corrent_sample_num


def test(model, dataloader, column, save_negative_samples = False, cb_id = 0):
    # import ipdb
    # ipdb.set_trace()
    dataset_columns = dataloader.dataset.features.keys()
    print(dataset_columns)

    negative_sample_output_filename = f'~/negative-{column}.txt'
    if os.path.exists(negative_sample_output_filename):
        os.remove(negative_sample_output_filename)

    all_indices = []
    all_labels = []
    total_num = 0
    for batch in tqdm(dataloader):
        embedding = batch[column].to("cuda:0")
        labels = batch['labels'] #这是真实的分类标签

        #返回的第一个参数是码本里的码，第二个参数是一级分类信息，第二个参数是二级分类信息
        indices = model.get_indices(embedding, False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for v in indices:
            all_indices.append(v)
        for v in labels.numpy():
            all_labels.append(v)

        total_num += indices.shape[0]


        if save_negative_samples:
            negative_indices = torch.nonzero((labels[:,0]==23) & (indices[:,0] == 1), as_tuple=True)[0].cpu().numpy()
            # print(f"There are {len(negative_indices)} samples should be in type-6 , but trained to type-1:")

            with open(negative_sample_output_filename, 'a', encoding='utf-8') as f:
                for id in negative_indices:
                    tmp = {}
                    tmp['label']=labels[id].tolist()
                    tmp['indices']=indices[id].tolist()
                    tmp['token'] = batch['token'][id]
                    f.write(json.dumps(tmp) + '\n')
        

    all_indices = np.vstack(all_indices)
    all_labels = np.vstack(all_labels)
    #import ipdb
    #ipdb.set_trace()
    res = convert_true_lable_to_cbindex(model.num_emb_list, all_indices, all_labels, cb_id = cb_id)

    right_nums = res['correct_sample_num']
    print(f"level0={right_nums[0]}/{total_num}  ({right_nums[0]/total_num:.4f}); level1={right_nums[1]} ({right_nums[1]/total_num:.4f})")


if __name__ == "__main__":
    local_args = parse_args()
    print(local_args)
    device = torch.device("cuda:0")

    ckpt = torch.load(local_args.ckpt_path, map_location=torch.device('cuda:0'))
    args = ckpt["args"]

    # 得到待处理的数据
    from dataprocess import data_process
    dataset, labels_mapping, label_dict = data_process(
                                            column_name=local_args.column_name,
                                            dataset_save_dir=local_args.dataset_save_dir)
    def remove_first_column(example):
        # 假设 'labels' 是一个二维张量，去掉第一列
        example['labels'] = example['labels'][1:]
        return example

    # 仅保留第一类food商品
    if args.dataset == "Amazon-531":
        lables_mapping={'level-0-mapping': {10: 0}, 11: {12: 0, 45: 1, 67: 2, 69: 3, 109: 4, 205: 5, 419: 6}, 44: {45: 6, 67: 7, 69: 8, 72: 9, 73: 10, 173: 11, 468: 12}, 54: {55: 12, 83: 13, 95: 14, 173: 15, 322: 16}, 60: {61: 18, 92: 19, 197: 20, 234: 21, 239: 22, 414: 23, 489: 24}, 64: {65: 24, 79: 25, 80: 26, 90: 27, 93: 28, 335: 29, 338: 30, 442: 31, 448: 32, 461: 33, 473: 34}, 220: {221: 30, 242: 31, 510: 32}, 10: {11: 0, 44: 1, 54: 2, 60: 3, 64: 4, 220: 5}}
        if args.filter_dataset_first_id != -1:
            dataset = dataset.map(remove_first_column)

    in_dim = len(dataset['train'][0][local_args.column_name])
    print(f"==============Data OK  in_dim={in_dim}===================================")

    if local_args.normal:
        model = RQVAE(in_dim=in_dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )
    else:
        model = RQVAE(in_dim=in_dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  architecture=args.architecture,
                  num_quantizers=args.num_quantizers,
                  last_codebook_shared=args.last_codebook_shared,
                  labels_mapping=labels_mapping,
                  init_codebook_with_description_embedding=args.init_codebook_with_description_embedding,
                  class_label_embedding_path=args.class_label_embedding_path,
                  init_class_embedding_codebook_level=args.init_class_embedding_codebook_level,
                  custom_loss_weight=args.custom_loss_weight,
                  custom_loss_type=args.custom_loss_type,
                  custom_codebook_residual=args.custom_codebook_residual,
                  custom_loss_weight_gamma=args.custom_loss_weight_gamma,
                  )
    print(model)

    # 从checkpoint中加载模型参数
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset['train'], shuffle=False, batch_size=args.batch_size)
    test(model, loader, local_args.column_name, save_negative_samples=local_args.save_negative_samples, cb_id = local_args.cb_id)
    # 通过split参数来确认待处理数据的数据是train/dev/test中的哪些
    # 分别生成对应的分类信息文件，
    # 如针对test数据集，基于epoch=999的模型参数，得到的文件是epoch_999_test.json
    #data_splits = local_args.split.split("|")
    #for split in data_splits:
    #    dataset = dataset.remove_columns(['id', 'extracted_keywords', 'original_token', 'embedding',  'embedding_original_token', 'embedding_extracted_keywords'])
    #    dataset[split].set_format('torch', columns=['labels', 'embedding_original_token_normalize', 'embedding_extracted_keywords_normalize', 'embedding_normalize'], output_all_columns=True)
    #    loader = DataLoader(dataset[split], num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    #    test(model, loader, split, save_negative_samples=local_args.save_negative_samples)
