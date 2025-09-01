import argparse
import random
import torch
import numpy as np
import logging

# from torch.utils.data import DataLoader

import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str,
                        default="../data/Games/Games.emb-llama-td.npy",
                        help="Input data path.")

    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", action='store_true', help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")
    parser.add_argument("--dataset", type=str, default="WebOfScience", help="WebOfScience/")
    parser.add_argument('--embedding_model', type=str, default='bert-base-uncased', help="bert-base-uncased / bge-large-en-v1.5 ")
    parser.add_argument('--init_method', type=str, help="full_init or load_from_ckpt")
    parser.add_argument('--use_category', action='store_true', help="This parameter used for Amazon-392 datasets to indicate read which json file")
    parser.add_argument('--save_negative_samples', action='store_true')
                        
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

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    from dataprocess import data_process
    class_dataset, labels_mapping, label_dict = data_process(
                                            args.dataset, 
                                            model_name=args.embedding_model,
                                            specific_file="/mnt/disk5/hewenting_nfs_serverdir/tmp/Amazon531-classname-level1.json")
    train_dataset, labels_mapping, label_dict = data_process(
                                            args.dataset, 
                                            model_name=args.embedding_model)
    in_dim = len(dataset['train'][0]['embedding'])
    print("==============Data OK===================================")

    in_dim = len(dataset['train'][0]['embedding'])
    latent = class_dataset['train']['embedding'].view(-1, in_dim)
    
    iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
    for batch_idx, data in enumerate(iter_data):
        embedding = data['embedding'].to(args.device)
        labels = data['labels']
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, embedding.t()) #d is a tensor, d.shape=[12101,256]

        indices = torch.argmin(d, dim=-1)