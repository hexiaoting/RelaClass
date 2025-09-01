# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
import argparse
import numpy as np

from torch.utils.data import DataLoader
import itertools
import json
import os
import tempfile

from models.rqvae import RQVAE

removed_first_label_flag = False
gt_labels = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--dataset_save_dir', type=str, default="/home_new/hewenting/embeddings-20250427/Amazon-hwt/train-task12_deduplicate_22068/")
    parser.add_argument('--normal', action='store_true', help="This is a normal rqvae model")
    parser.add_argument('--cb_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='wos')
    # parser.add_argument('--save_negative_samples', action='store_true', help="如果设置，则可以将错误的样本打印出来，但是需要改trainer.py里的negative_indices")
    return parser.parse_args()

codebook_shared_status=[False, False, False]
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
    print("level=",level, "   lableid2idx_mapping=",lableid2idx_mapping)
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
        if level == 0 and i != 1:
            continue
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
                if aa[0] in correct_samples_num_1st.keys():
                    import ipdb
                    ipdb.set_trace()
                    print(level, correct_samples_num_1st)
                correct_samples_num_1st[aa[0]] = aa[1]
                print("\t\t\t\t\t\t\t\t\tcorrect_samples_num_1st=", correct_samples_num_1st)
        
    res = {}
    res['new_label'] = new_label
    res['distinct_class_num'] = [len(correct_samples_num_1st.keys()), distinct_class_num_2nd, distinct_class_num_3rd]
    res['correct_sample_num'] = [sum(correct_samples_num_1st.values()), correct_samples_num_2nd, correct_samples_num_3rd]
    res['codebook_order'] = [codebook_order_1st, codebook_order_2nd, codebook_order_3rd]
    return res

def convert_true_lable_to_cbindex2(indices, labels, level=0):
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


def calculate_ranks_from_similarities(all_similarities, positive_relations):
    """
    all_similarities: a np array 【.shape=(分类个数,)】
    positive_relations: a list of array indices    [positive_relations=[2,17]]

    return a list
    """
    # positive_similarities = all_similarities[positive_relations]
    # # 构造负样本（mask掉正样本）
    # negative_mask = np.ones_like(all_similarities, dtype=bool)
    # negative_mask[positive_relations] = False
    # negative_similarities = all_similarities[negative_mask]

    # # 计算每个正样本与所有负样本的比较：有多少负样本 >= 正样本
    # # 这里使用广播机制：(N_positive, N_negative)
    # ranks = (negative_similarities <= positive_similarities[:, np.newaxis]).sum(axis=1) + 1

    # return list(ranks)
    
    positive_relation_similarities = all_similarities[positive_relations]
    negative_relation_similarities = np.ma.array(all_similarities, mask=False)
    negative_relation_similarities.mask[positive_relations] = True
    ranks = (negative_relation_similarities <= positive_relation_similarities[:, np.newaxis]).data.sum(axis=1).tolist()
    return ranks

def mrr(all_ranks):
    """ Scaled MRR score, check eq. (2) in the PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf
    """
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions)
    return (1.0 / scaled_rank_positions).mean()


def example_f1(trues, preds):
    """
    trues: a list of true classes
    preds: a list of model predicted classes
    """
    print("          Note: for wos dataset, we add second lable id += 7")
    f1_list = []
    for t, p in zip(trues, preds):
        t[1] += 7
        p[1] += 7
        f1 = 2 * len(set(t) & set(p)) / (len(t) + len(p))
        f1_list.append(f1)
    return np.array(f1_list).mean()


def precision_at_k(preds, gts, k=1, mapping={}):
    assert len(preds) == len(gts), "number of samples mismatch"
    if k > len(gts):
        return None
    
    p_k = 0.0
    if removed_first_label_flag and k == 2:
        for pred, gt in zip(preds, gts):
            if pred[0] == gt[1]:
                p_k += 1
    elif removed_first_label_flag and k == 1:
        for pred, gt in zip(preds, gts):
            if mapping[pred[0]] == gt[0]:
                p_k += 1
    
    else:
        for pred, gt in zip(preds, gts):
            if k == 1 and pred[0] == gt[0]:
                p_k += 1
            elif k == 2 and pred[0] == gt[0] and pred[1] == gt[1]:
                p_k += 1
            else:
                if k == 2 and pred[1]==gt[1]:
                    print(pred, gt)
                    import ipdb
                    ipdb.set_trace()


    p_k /= len(preds)
    return p_k


def get_map_from_secondlevel_to_firstlevel(labels_mapping):
    mapping = {}
    class_l1 = labels_mapping['level-0-mapping'].keys()
    for l1 in class_l1:
        for l2 in labels_mapping[l1]:
            mapping[l2] = l1
            
    return mapping

def reconstruct_and_flat_codebook(model, dataset):
    weight = model.rq.vq_layers[0].embedding.weight
    codebook = None
    
    if dataset == "wos":
        groups = [5, 3, 5, 5, 5, 5, 5]
        # 分段取平均
        start = 0
        averaged_vectors = []
        for size in groups:
            end = start + size
            segment = weight[start:end]  # 取出当前段 [size, 32]
            print(start, "\t", end, '\t',segment.shape)
            mean_vec = segment.mean(dim=0, keepdim=True)  # [1, 32]
            averaged_vectors.append(mean_vec)
            start = end
            
        averaged_result = torch.cat(averaged_vectors, dim=0)
        codebook = torch.cat([averaged_result, weight], dim=0) 
        print(codebook.shape, codebook[7]==weight[0])
    else:
        raise Exception("Todo")

    return codebook

def test(model, dataset, dataloader, column, gt_labels, labels_mapping, cb_id = 0):
    all_indices = []
    all_labels = []
    all_ids = []
    all_prediction = []
    all_ranks = []
    total_num = 0
    
    # import ipdb
    # ipdb.set_trace()
    
    # asin_2_index={}
    index2asin={}
    # codebooks = reconstruct_and_flat_codebook(model, dataset)
    wos_cb21_mapping = {}
    for batch in tqdm(dataloader):
        embedding = batch[column].to("cuda:0")
        labels = batch['labels'] #这是真实的分类标签
        ids = batch['asin']

        #返回的第一个参数是码本里的码，第二个参数是一级分类信息，第二个参数是二级分类信息
        if dataset=='wos':
            prediction_tmp = model.get_distance(embedding, cb_id, need_inference_first_class=remove_first_column, labels_mapping=labels_mapping)
            
            for label in gt_labels[:,:2].numpy():
                if label[1] not in wos_cb21_mapping.keys():
                    wos_cb21_mapping[label[1]] = label[0]
        else:
            tmp = model.get_distance(embedding, cb_id, need_inference_first_class=False, labels_mapping=labels_mapping)
            prediction_tmp = torch.cat([i for i in tmp], dim=1)
        all_prediction.append(prediction_tmp)
        
        indices = model.get_indices(embedding, False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        
        for i, value in enumerate(indices):
            v = value.tolist()[:(cb_id+1)]
            all_indices.append(value)
            # asin_2_index[batch['asin'][i]] = v
            if v[-1] not in index2asin.keys():
                index2asin[v[-1]] = []
            index2asin[v[-1]].append(batch['asin'][i])
        for v in labels.numpy():
            all_labels.append(v)
        for id in ids:
            all_ids.append(id)

        total_num += indices.shape[0]
        
        
    #     encoder_embedding = model.encoder(embedding).detach()
    #     d = torch.sum(encoder_embedding**2, dim=1, keepdim=True) + \
    #         torch.sum(codebooks**2, dim=1, keepdim=True).t()- \
    #         2 * torch.matmul(encoder_embedding, codebooks.t())
    #     for i, label in enumerate(labels.numpy()):
    #         convert_label = [wos_cb21_mapping[label[0]],  label[0] + 7]
    #         all_ranks.append(calculate_ranks_from_similarities(d[i].detach().cpu().numpy(), convert_label))
        
        
    # print(f"MRR: {mrr(all_ranks)}")
    # with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
    #     json.dump(asin_2_index, tmp_file)
    #     temp_filename = tmp_file.name  # 保存文件名，以便后续使用
    # print(f"asin_2_index临时文件已创建: {temp_filename}")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(index2asin, tmp_file)
        temp_filename = tmp_file.name  # 保存文件名，以便后续使用
    print(f"index2asin临时文件已创建: {temp_filename}")

    all_indices = np.vstack(all_indices)
    all_labels = np.vstack(all_labels)
    # mapping_tmp = {}
    # for _,v in labels_mapping.items():
    #     for a,b in v.items():
    #         if a in mapping_tmp.keys():
    #             print("error")
    #             import ipdb
    #             ipdb.set_trace()
    #         mapping_tmp[a] = b
    first_level_class_num = len(labels_mapping['level-0-mapping'].keys())
    # import ipdb
    # ipdb.set_trace()
    for i, label in enumerate(all_labels):
        gold_label = []
        predict_label = []
        if dataset=='wos':
            gold_label = gt_labels[i].numpy()[:2].tolist()
            gold_label[1] += 7
            if removed_first_label_flag:
                predict_label.append(wos_cb21_mapping[all_indices[i][0]])
                predict_label.append(all_indices[i][0] + 7)
                print(f"id:{all_ids[i]},gold_label:{gold_label}, predict_label:{predict_label}")
            else:
                predict_labels = all_indices[i][:2]
                for l in predict_labels:
                    l[1] += 7
                print(f"id:{all_ids[i]},gold_label:{gold_label}, predict_label:{all_indices[i][:2]}")
                
        else:
            for j, id in enumerate(label):
                # gold_label.append(mapping_tmp[id])
                if j == 0:
                    gold_label.append(labels_mapping['level-0-mapping'][id])
                    predict_label.append(all_indices[i][0])
                elif j == 1:
                    gold_label.append(labels_mapping[label[0]][id] + first_level_class_num)
                    predict_label.append(all_indices[i][1] + first_level_class_num)
                elif j == 2:
                    gold_label.append(labels_mapping[label[1]][id] + 5 + 52)
                    predict_label.append(all_indices[i][2] + 5 + 52)
            # print(f"gold_label:{gold_label}, predict_label:{predict_label}")

        
    mrr(all_ranks)

    flattened = [tensor for sublist in all_prediction for tensor in sublist]
    all_prediction = torch.cat(flattened, dim=0).cpu()
    
    # import ipdb
    # ipdb.set_trace()
    cb_size = len(all_labels[0])
    res = convert_true_lable_to_cbindex(model.num_emb_list, all_indices[:,:cb_size], all_labels)

    right_nums = res['correct_sample_num']
    print(f"level0={right_nums[0]}/{total_num}  ({right_nums[0]/total_num:.4f}); level1={right_nums[1]} ({right_nums[1]/total_num:.4f}); level2={right_nums[2]} ({right_nums[2]/total_num:.4f})")
    # exit()
    
    if np.all(all_labels[:, -1] == -1): #最后一列都是-1就不要这一列了
        all_labels = all_labels[:, 0]
        all_labels = all_labels[:, np.newaxis]
        gt_labels = gt_labels[:,:-1].tolist()
    all_ranks = []
    top_classes = []
    size = len(all_labels[0])
    for pred, gt in zip(all_prediction, gt_labels):
        rank1 = calculate_ranks_from_similarities(pred[:7], [gt[0]])[0].item()
        rank2 = calculate_ranks_from_similarities(pred[7:], [gt[1]])[0].item()
        all_ranks.append([rank1, rank2])
        # all_ranks.append(calculate_ranks_from_similarities(pred, gt))
        top_classes.append(np.argsort(pred[7:])[:size].tolist())
    
    mapping = None
    if remove_first_column:
        mapping = get_map_from_secondlevel_to_firstlevel(labels_mapping)
        for idx, cls in enumerate(top_classes):
            top_classes[idx] = np.insert(top_classes[idx],0,mapping[cls[0]] )
            
    for k in [1, 2, 3]:
        print(f"Precision@{k}: {precision_at_k(top_classes, gt_labels, k, mapping)}")
    print(f"MRR: {mrr(all_ranks)}")
 
    print(f"Exmaple F1: {example_f1(gt_labels, top_classes)}") 
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    # import ipdb
    # ipdb.set_trace()
    local_args = parse_args()
    print(local_args)
    device = torch.device("cuda:0")

    ckpt = torch.load(local_args.ckpt_path, map_location=torch.device('cuda:0'))
    args = ckpt["args"]

    # 得到待处理的数据
    from dataprocess import data_process
    dataset, labels_mapping, label_dict = data_process(
                                            column_name=args.column_name,
                                            dataset_save_dir=local_args.dataset_save_dir)
    gt_labels = dataset['train']['labels']
    
    def remove_first_column(example):
        # 假设 'labels' 是一个二维张量，去掉第一列
        example['labels'] = example['labels'][1:]
        return example

    removed_first_label_flag = False
    if args.filter_dataset_first_id != -1:
        removed_first_label_flag = True
        dataset = dataset.map(remove_first_column)

    in_dim = len(dataset['train'][0][args.column_name])
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
    test(model, args.dataset, loader, args.column_name, gt_labels, labels_mapping, cb_id = local_args.cb_id)


#python3.8  inference.py --ckpt_path  /home_new/hewenting/checkpoint_gpu9/webofsicence/wos11967/train_test/nokeywords-task0__fullInit_num60-task0__precision0.6959/epoch_1379_precision_0.6959_0.0000_0.0000_model.pth --dataset_save_dir /home_new/hewenting/embeddings-20250427/WebOfScience/wos11967-train-test/wos_nokeywords_test_task0

#python3.8  inference.py --ckpt_path  /home_new/hewenting/checkpoint_gpu9/hwt-dataset-train-test/fix_12cb_train3cb/Aug-17-2025_22-54-36/epoch_169_precision_0.9371_0.8542_0.7071_model.pth --dataset_save_dir /home_new/hewenting/embeddings-20250427/Amazon-hwt/test-task12/