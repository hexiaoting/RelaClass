import argparse
import random
import torch
import numpy as np
import logging
import os

from torch.utils.data import DataLoader

# from emb_dataset import EmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=10, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str,
                        default="../data/Games/Games.emb-llama-td.npy",
                        help="Input data path.")

    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", action='store_true', help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--custom_loss_weight', type=float, default=0, help='vq quantion diversity  or first cb similarity loss weight')
    parser.add_argument("--custom_loss_type", type=str, default="l1", help="custom_loss_type")
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint file")
    parser.add_argument("--dataset", type=str, default="WebOfScience", help="WebOfScience/")
    parser.add_argument('--embedding_model', type=str, default='SFR-Embedding-2_R', help="bert-base-uncased / bge-large-en-v1.5 / SFR-Embedding-2_R / bge-en-icl ")
    parser.add_argument('--init_method', type=str, help="full_init or load_from_ckpt or load_from_normalrqvae_ckpt")
    parser.add_argument('--codebook1_order', type=int, nargs='+', required=False, help='If init_method is load_from_normalrqvae_ckpt, then we need to know the first codebook index order')
    parser.add_argument('--use_category', action='store_true', help="This parameter used for Amazon-392 datasets to indicate read which json file")
    parser.add_argument('--save_positive_samples', action='store_true', help="如果设置，则可以将错误的样本打印出来，但是需要改trainer.py里的negative_indices")
    parser.add_argument('--save_negative_samples', action='store_true', help="如果设置，则可以将错误的样本打印出来，但是需要改trainer.py里的negative_indices")
    parser.add_argument('--negative_condition', type=int, nargs='+',default=[23, 2] , help="3 2表示把第3类商品分到了码号为2的上面")
    parser.add_argument('--specific_datafile', type=str, help="如果指定specific_file，则会专门处理这个json文件的内容进行embedding，基于该数据再去训练模型", default=None)
    
    parser.add_argument('--architecture', type=str, default='tree_residual_uniform')
    parser.add_argument('--num_quantizers',  type=int,  default=3)
    parser.add_argument('--filter_dataset_first_id',  type=int,  default=-1)
    parser.add_argument('--last_codebook_shared', action='store_true', help="如果设置，则表示最后一个码本是共享的")
    parser.add_argument('--column_name', type=str, default='embedding_normalize')
    parser.add_argument('--dataset_save_dir', type=str, help='dataset saving dir, like /home/hewenting/data_preprocess/Amazon-531/SFR-Embedding-2_R_28932_merged')
    parser.add_argument('--init_codebook_with_description_embedding', action='store_true', help="如果设置，则表明用第一级大类的类标签体系进行embedding后的数据进行初始化第一个码本")
    parser.add_argument('--init_class_embedding_codebook_level', type=int,  nargs='+', default=[0])
    parser.add_argument('--class_label_embedding_path', type=str, nargs='+', default=None)
    parser.add_argument('--custom_codebook_residual', type=int, default=0, help="这个参数只有init_class_embedding_codebook_level>0的时候才有效，0表示不需要将码本减去前续码本做残差再进行loss计算，1表示要减去前续的码本信息再做f1 loss")
    parser.add_argument('--custom_loss_weight_gamma', type=float, default=1)
    parser.add_argument('--target_codebook_generate_method', type=str, default="no", help="user means using user's local file, system means using 样本自己的计算出来的")
    parser.add_argument('--target_codebook_system_datasets', type=str, default=None, help="用大模型标注出来的正确的样本计算出来的数据集embedding")
    parser.add_argument('--filter_range_cb3rd', type=int, nargs='+', default=[])

    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")
    if args.init_codebook_with_description_embedding and (args.class_label_embedding_path is None and args.target_codebook_generate_method == 'user'):
        raise Exception(f"You want to init the first codebook embedding, but donot set init_class_desciption_embedding_path")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    # hwt修改为：从预设的目录读取，或从原始的json文件读取后进行embedding存储到固定的目录
    # 由于SFR、BGE等模型在embedding时，会进行normalize，所以我生成了embedding和embedding_normalize的列信息分别代表编码时normalize前后的embedding。
    #     默认用"embedding_normalize"（这个效果比embedding要好）
    from dataprocess import data_process, generate_mapping
    # column_name = "embedding_normalize"
    # import ipdb
    # ipdb.set_trace()
    dataset, labels_mapping, label_dict = data_process(
                                            args.dataset,
                                            column_name=args.column_name,
                                            dataset_save_dir=args.dataset_save_dir)

    # import ipdb
    # ipdb.set_trace()

    def remove_first_column(example):
        # 假设 'labels' 是一个二维张量，去掉第一列
        example['labels'] = example['labels'][1:]
        return example

    # 仅保留第一类food商品
    if args.dataset == "Amazon-531":
        if args.filter_dataset_first_id != -1:
            dataset = dataset.map(remove_first_column)
    #        if len(labels_mapping['level-0-mapping'].keys()) == 1 and 169 in labels_mapping['level-0-mapping'].keys():
    #            labels_mapping['level-0-mapping'] = labels_mapping[169]
    #        
    dataset_test = None
    if args.dataset == "hwt-dataset" :
        print(len(labels_mapping['level-0-mapping'].keys()))
        print(' '.join(str(len(labels_mapping[k2].keys())) for k2 in labels_mapping['level-0-mapping'].keys()))
        for k1 in labels_mapping['level-0-mapping'].keys(): #{0: 0, 48: 1, 117: 2, 190: 3, 352: 4} 
            for k2 in labels_mapping[k1].keys(): #0: {1: 0, 3: 1, 6: 2, 15: 3, 18: 4, 20: 5}
                print(len(labels_mapping[k2].keys()), end=' ')
        print()
        test_datapath = "/home_new/hewenting/embeddings-20250427/Amazon-hwt/test-task12/"
        dataset_test, _, _ = data_process(
                                            args.dataset,
                                            model_name=args.embedding_model,
                                            with_category=args.use_category,
                                            specific_file=args.specific_datafile,
                                            column_name=args.column_name,
                                            dataset_save_dir=test_datapath)
    if args.dataset == "wos":
        test_datapath = "/home_new/hewenting/embeddings-20250427/WebOfScience/wos11967-train-test/wos_nokeywords_test_task0/"
        dataset_test, _, _ = data_process(
                                            args.dataset,
                                            model_name=args.embedding_model,
                                            with_category=args.use_category,
                                            specific_file=args.specific_datafile,
                                            column_name=args.column_name,
                                            dataset_save_dir=test_datapath)
        # import ipdb
        # ipdb.set_trace()
        # convert_mapping = {"3-13":[7,100,-1],"3-14":[8,101,-1], "3-16":[9,102,-1], "4-18":[10,103,-1],"4-20":[11,104,-1],"4-22":[12,105,-1]}
        convert_mapping = {"3-13":[0,100,-1],"3-14":[7,101,-1], "3-16":[1,102,-1], "4-18":[0,103,-1],"4-20":[0,104,-1],"4-22":[8,105,-1]}
        # 8 4 5 3 3 5 5 1 1
        def tmp_convert_labels_testing(example):
            id = '-'.join(str(l) for l in example['labels'][:2].tolist())
            if id in convert_mapping.keys():
                example['labels'] = torch.tensor(convert_mapping[id])
            return example
        dataset = dataset.map(tmp_convert_labels_testing)
        dataset_test = dataset_test.map(tmp_convert_labels_testing)
        labels_mapping = generate_mapping(dataset['train']['labels'][:,:2], process_3rd_mapping=False)
        print(labels_mapping)
        # labels_mapping = {'level-0-mapping': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7:7, 8:8}, 0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 100:5, 103:6, 104:7}}
    
        if args.filter_dataset_first_id != -1:
            dataset = dataset.map(remove_first_column)
            dataset_test = dataset_test.map(remove_first_column)
            sorted_values = dataset['train']['labels'][:,0].unique().tolist()
            labels_mapping['level-0-mapping'] = {}
            for idx, v in enumerate(sorted_values):
                labels_mapping['level-0-mapping'][idx] = v
    # if args.dataset == "hwt-dataset":
    #     def filter_hwtdataset_examples(example):
    #         candidate = example['labels'][2]
    #         if not isinstance(example['labels'], list):
    #             candidate = candidate.item()
            
    #         return candidate not in [113, 94]
    #     dataset = dataset.filter(filter_hwtdataset_examples)
    #     print(f"remove some samples, now we have {dataset['train'].shape[0]}")
        
        # l1 = 0
        # l2 = -1
        # def group_hwtdataset_samples(example):
        #     labels = example['labels']
        #     if l2 == -1:
        #         return labels[0] == l1
        #     else:
        #         return labels[0] == l1 and labels[1] == l2
        
        # target_cb_embeddings_from_traindata_1st = []
        # target_cb_embeddings_from_traindata_2nd = []
        # if not os.path.exists('/home/hewenting/data_preprocess/hwt-dataset-20250410/all/corpus_codebook/tensor_list_1st.pt'):
        #     for k1 in labels_mapping['level-0-mapping'].keys():
        #         l1 = k1
        #         l2 = -1
        #         tmp_dataset = dataset.filter(group_hwtdataset_samples)
        #         target_cb_embeddings_from_traindata_1st.append(tmp_dataset['train'][args.column_name])
        #         for k2 in labels_mapping[k1].keys():
        #             l2 = k2
        #             if k1 == 48 and k2 == 112:
        #                 continue
        #             target_cb_embeddings_from_traindata_2nd.append(tmp_dataset.filter(group_hwtdataset_samples)['train'][args.column_name])
        #     torch.save(target_cb_embeddings_from_traindata_1st, '/home/hewenting/data_preprocess/hwt-dataset-20250410/all/corpus_codebook/tensor_list_1st.pt')
        #     torch.save(target_cb_embeddings_from_traindata_2nd, '/home/hewenting/data_preprocess/hwt-dataset-20250410/all/corpus_codebook/tensor_list_2nd.pt')
        # else:
        #     target_cb_embeddings_from_traindata_1st = torch.load('/home/hewenting/data_preprocess/hwt-dataset-20250410/all/corpus_codebook/tensor_list_1st.pt')
        #     target_cb_embeddings_from_traindata_2nd = torch.load('/home/hewenting/data_preprocess/hwt-dataset-20250410/all/corpus_codebook/tensor_list_2nd.pt')
            
    # print(dataset['train'].features.keys()) #['extracted_keywords', 'original_token', 'token', 'id', 'labels', 'embedding_extracted_keywords', 'embedding_extracted_keywords_normalize', 'embedding', 'embedding_normalize']
    in_dim = len(dataset['train'][0][args.column_name])
    print(f"==============Data OK  in_dim={in_dim}===================================")
    # import ipdb
    # ipdb.set_trace()

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
                  filter_range_cb3rd = args.filter_range_cb3rd,
                  )
    print(model)
    # hwt修改
    # def filter_labels(example):
    #     # 检查 'labels' 过滤出beauty和health_personal_care类型的数据
    #     return example['labels'][0] in [10,23]

    # filtered_dataset = dataset['train'].filter(filter_labels)
    # data_loader = DataLoader(filtered_dataset,num_workers=args.num_workers,
    data_loader = DataLoader(dataset['train'],num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    

    test_data_loader = DataLoader(dataset_test['train'],num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args,model, len(data_loader))
    best_loss, best_precision_rate, best_secondlevel_precision, best_thirdlevel_precision = trainer.fit(data_loader,test_data_loader, args.column_name, labels_mapping)

    print("Best Loss",best_loss)
    print(f"Best Precision Rate {best_precision_rate} / {best_secondlevel_precision} / {best_thirdlevel_precision}")

