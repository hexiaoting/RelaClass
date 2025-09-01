# import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F
import datasets
import random
import numpy

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 architecture='',
                 num_quantizers=3,
                 last_codebook_shared=False,
                 labels_mapping={},
                 init_codebook_with_description_embedding=False,
                 class_label_embedding_path=None,
                 init_class_embedding_codebook_level=[0],
                 custom_loss_weight=0,
                 custom_loss_type='mse', 
                 custom_codebook_residual=0,    
                 custom_loss_weight_gamma=1,  
                 filter_range_cb3rd=[],
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.architecture = architecture
        self.num_quantizers = num_quantizers
        self.last_codebook_shared = last_codebook_shared
        self.labels_mapping = labels_mapping
        self.init_codebook_with_description_embedding = init_codebook_with_description_embedding
        self.class_label_embedding_path = class_label_embedding_path
        self.init_class_embedding_codebook_level = init_class_embedding_codebook_level
        self.custom_loss_weight = custom_loss_weight
        self.custom_loss_type = custom_loss_type
        self.custom_codebook_residual = custom_codebook_residual
        self.custom_loss_weight_gamma = custom_loss_weight_gamma

        self.target_corpus_embeddings_from_traindata_1st = []#target_corpus_embeddings_from_traindata_1st
        self.target_corpus_embeddings_from_traindata_2nd = []#target_corpus_embeddings_from_traindata_2nd
        self.target_corpus_embeddings_from_traindata_3rd = []
        self.imported_embedding_full_cuda = [None, None, None]
        self.target_codebook_embedding_1st = None
        self.target_codebook_embedding_2nd = None
        self.target_codebook_embedding_3rd = None
        self.need_encoder_for_target_embedding = False
        self.filter_range_cb3rd = filter_range_cb3rd
        
        

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          architecture=self.architecture,
                                          num_quantizers=self.num_quantizers,
                                          labels_mapping = self.labels_mapping,
                                          last_codebook_shared = self.last_codebook_shared,
                                          init_codebook_with_description_embedding=self.init_codebook_with_description_embedding,
                                          init_class_embedding_codebook_level=self.init_class_embedding_codebook_level)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        
      
        

    def init_target_cb_embeddding(self, loader, labels_mapping, column_name, 
                                  ckpt_dir, 
                                  using_upload_file, 
                                  process_cb_idx=[], 
                                  target_datasets_path=None,
                                  remove_first_label=False,):
        if using_upload_file:  # 用用户自己生成的embedding初始化
            assert(self.init_codebook_with_description_embedding)
            assert(self.init_class_embedding_codebook_level == process_cb_idx)
            if process_cb_idx == [0]:
                self.imported_embedding_full_cuda[0] = torch.load(self.class_label_embedding_path[0]).cuda()
            elif process_cb_idx == [1]:
                self.imported_embedding_full_cuda[1] = torch.load(self.class_label_embedding_path[0]).cuda()
            elif process_cb_idx == [2]:
                self.imported_embedding_full_cuda[2] = torch.load(self.class_label_embedding_path[0]).cuda()
            elif process_cb_idx == [0, 1]:
                self.imported_embedding_full_cuda[0] = torch.load(self.class_label_embedding_path[0]).cuda()
                self.imported_embedding_full_cuda[1] = torch.load(self.class_label_embedding_path[1]).cuda()
            elif process_cb_idx == [0, 1, 2]:
                self.imported_embedding_full_cuda[0] = torch.load(self.class_label_embedding_path[0]).cuda()
                self.imported_embedding_full_cuda[1] = torch.load(self.class_label_embedding_path[1]).cuda()
                self.imported_embedding_full_cuda[2] = torch.load(self.class_label_embedding_path[2]).cuda()
            else:
                raise Exception("Failed.")
            if 0 in process_cb_idx:
                self.need_encoder_for_target_embedding = True
                return
        else:
            cb1_output_file = os.path.join(ckpt_dir, "../corpus_all_codebook1_target_embedding.pt")
            cb2_output_file = os.path.join(ckpt_dir, "../corpus_all_codebook2_target_embedding.pt")
            cb3_output_file = os.path.join(ckpt_dir, "../corpus_all_codebook3_target_embedding.pt")

            l1 = 0
            l2 = -1
            l3 = -1
            def group_hwtdataset_samples(example):
                labels = example['labels']
                start = 0
                if remove_first_label:
                    start = 1
                if l2 == -1:
                    return labels[0 + start] == l1
                elif l3 == -1:
                    return labels[0 + start] == l1 and labels[1 + start] == l2
                else:
                    return labels[0 + start] == l1 and labels[1 + start] == l2 and labels[2 + start] == l3

            if target_datasets_path is not None or ((0 in process_cb_idx and not os.path.exists(cb1_output_file)) or \
                (1 in process_cb_idx and not os.path.exists(cb2_output_file)) or \
                (2 in process_cb_idx and not os.path.exists(cb3_output_file))):
                
                if target_datasets_path is not None:
                    dss = datasets.load_from_disk(target_datasets_path)
                    

                    dss['train'].set_format('torch', columns=['labels'], 
                                            output_all_columns=True)
                    dss['train'].set_format('torch', columns=[column_name], 
                                            output_all_columns=True,
                                            format_kwargs={"dtype": torch.float32})
                    
                    # def convert_to_tensor(example):
                    #     with open("tttttttt", 'a+') as f:
                    #         labels = example['labels'].tolist()  # 转为 list
                    #         token = example['token']
                    #         line = f"Labels: {labels}, Token: {token}\n"
                    #         f.write(line)
                    #     print(example['labels'], end=' ')
                    #     example['labels'] = torch.tensor(example['labels'], dtype=torch.int)
                    #     print("ok")
                    #     return example

                    # ds = dss['train'].map(convert_to_tensor, num_proc=1)
                    
                    ds = dss['train']
                    # if remove_first_label:
                    #     def remove_first_column(example):
                    #         print(example['labels'], type(example['labels']))
                    #         # 假设 'labels' 是一个二维张量，去掉第一列
                    #         # if isinstance(dss['train'][0]['labels'], list):
                    #         #     example['labels'] = example['labels'][1:]
                    #         example['labels'] = example['labels'][1:]
                    #         return example
                    #     ds = ds.map(remove_first_column, num_proc=1)
                else:
                    ds = loader.dataset
                
                for k1 in labels_mapping['level-0-mapping'].keys():
                    l1 = k1
                    l2 = -1
                    # random_indices = random.sample(range(len(ds)), 1000)
                    # subset_ds = ds.select(random_indices)
                    # tmp_dataset = subset_ds.filter(group_hwtdataset_samples)
                    tmp_dataset = ds.filter(group_hwtdataset_samples)
                    
                    # if isinstance(tmp_dataset[column_name], numpy.ndarray):
                    #     self.target_corpus_embeddings_from_traindata_1st.append(torch.from_numpy(tmp_dataset[column_name]))
                    # else:
                    self.target_corpus_embeddings_from_traindata_1st.append(tmp_dataset[column_name])
                    
                    if process_cb_idx == [0]:
                        print("xxxxxxxx:", self.target_corpus_embeddings_from_traindata_1st[-1].shape)
                        if self.target_corpus_embeddings_from_traindata_1st[-1].shape[0] == 0:
                        continue
                    for k2 in labels_mapping[k1].keys():
                        l2 = k2
                        l3 = -1
                        self.target_corpus_embeddings_from_traindata_2nd.append(tmp_dataset.filter(group_hwtdataset_samples)[column_name])
                        if 2 not in process_cb_idx:
                            print("xxxxxxxx:", self.target_corpus_embeddings_from_traindata_2nd[-1].shape)
                            continue
                        for k3 in labels_mapping[k2].keys():
                            l3 = k3
                            self.target_corpus_embeddings_from_traindata_3rd.append(tmp_dataset.filter(group_hwtdataset_samples)[column_name])  
                            print("xxxxxxxx:", self.target_corpus_embeddings_from_traindata_3rd[-1].shape)

                if 0 in process_cb_idx:
                    self.need_encoder_for_target_embedding = True
                    return
                
            else:
                if 1 in process_cb_idx:
                    self.target_corpus_embeddings_from_traindata_2nd = torch.load(cb2_output_file)
                if 2 in process_cb_idx:
                    self.target_corpus_embeddings_from_traindata_3rd = torch.load(cb3_output_file)
                if 0 in process_cb_idx:
                    self.target_corpus_embeddings_from_traindata_1st = torch.load(cb1_output_file)
                    self.need_encoder_for_target_embedding = True
                    return
                    
                    
        
        # 如果是固定住第一个码本，则针对第二个码本的目标embedding，判断是否要计算差值（即减去第一个码本embedding）
        if process_cb_idx == [1] or process_cb_idx == [2]:
            if using_upload_file:
                target_codebook_embedding = self.encoder(self.imported_embedding_full_cuda[process_cb_idx[0]])
            else:
                tmp_corpus_embedding = self.target_corpus_embeddings_from_traindata_2nd if process_cb_idx == [1] else self.target_corpus_embeddings_from_traindata_3rd
                
                #TODO 这两个torch.mean有什么区别呢？
                if len(tmp_corpus_embedding) != 265 and len(tmp_corpus_embedding) != 52:
                    mean_tensors = [ torch.mean(self.encoder(class_l1.cuda()), dim=0) for class_l1 in tmp_corpus_embedding ]
                else:
                    mean_tensors = []
                    for _, class_l1 in enumerate(tmp_corpus_embedding):
                        aa = torch.mean(self.encoder(class_l1.cuda()), dim=0)
                        mean_tensors.append(aa)
                target_codebook_embedding = torch.stack(mean_tensors).cuda()
                
            if process_cb_idx == [1]:
                self.target_codebook_embedding_2nd  = target_codebook_embedding
            elif process_cb_idx == [2]:
                self.target_codebook_embedding_3rd = target_codebook_embedding
                
            if self.custom_codebook_residual == 0:
                return
            
            #计算残差
            cb1_weights = self.rq.vq_layers[0].embedding.weight
            
            if process_cb_idx == [1]:
                cb1_code_num = self.rq.vq_layers[1].n_e
                new_tensor = torch.zeros(cb1_code_num, self.e_dim)
                start_idx = 0
                for i, count in enumerate(self.num_emb_list[1 : (1 + self.num_emb_list[0])]):
                    end_idx = start_idx + count
                    new_tensor[start_idx : end_idx, :] = cb1_weights[i].unsqueeze(0).expand(count, -1)
                    start_idx = end_idx
            
                self.target_codebook_embedding_2nd -= new_tensor.detach().cuda()
                return
            
            elif process_cb_idx == [2]:
                cb2_weights = self.rq.vq_layers[1].embedding.weight
                cb0_code_num = self.rq.vq_layers[0].n_e
                cb1_code_num = self.rq.vq_layers[1].n_e
                cb2_code_num = self.rq.vq_layers[2].n_e
                new_tensor = torch.zeros(cb2_code_num, self.e_dim)
                start_idx = 0

                print(self.num_emb_list)
                for i in range(cb0_code_num):#0 1 2 3 4
                    for j in range(self.num_emb_list[1 + i]): #j 0 1 2 3 4 5 / 0 1 2 3 
                        count = self.num_emb_list[1 + cb0_code_num + j + sum(self.num_emb_list[1: 1+i])]
                        print(count, end=', ')
                        # print(f"i={i}, j={j}, count={count}")
                        end_idx = start_idx + count
                        new_tensor[start_idx : end_idx, :] = cb2_weights[j + sum(self.num_emb_list[1: 1+i])].unsqueeze(0).expand(count, -1) + cb1_weights[i]
                        start_idx = end_idx

                #########TODO only for beauty testing
                if len(self.filter_range_cb3rd) == 2 and (new_tensor.shape[0] > (self.filter_range_cb3rd[1] - self.filter_range_cb3rd[0])):
                    new_tensor = new_tensor[self.filter_range_cb3rd[0] : self.filter_range_cb3rd[1]]
                if len(self.filter_range_cb3rd) == 2 and (self.target_codebook_embedding_3rd.shape[0] > (self.filter_range_cb3rd[1] - self.filter_range_cb3rd[0])):
                    self.target_codebook_embedding_3rd = self.target_codebook_embedding_3rd[self.filter_range_cb3rd[0] : self.filter_range_cb3rd[1]]
                self.target_codebook_embedding_3rd -= new_tensor.detach().cuda()
            

    def vq_initialization(self, labels, embeddings, first_cb_is_initted=False, using_kmeans=False): #x.shape=[12101, 768]
        print("------->vq_initialization")
        encoded_embedding = self.encoder(embeddings)
        if self.architecture == "normal" or using_kmeans:
            self.rq.vq_normal_init(labels, encoded_embedding)
        elif self.architecture == "tree_residual_uniform":
            self.rq.residual_tree_init(labels, encoded_embedding, code_idx_list=[0], is_initted=first_cb_is_initted)
        else:
            raise Exception(f"{self.architecture} is not supported!")
        print("<-------vq_initialization")

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    @torch.no_grad()
    def get_distance(self, xs, codebook_id=0, need_inference_first_class=False, labels_mapping={}):
        x_e = self.encoder(xs)
        return  self.rq.get_distance(x_e, codebook_id, need_inference_first_class, labels_mapping)
    
    def custom_loss(self, codebook_level, expected_codebook_embedding):
        custom_loss = None
        if self.custom_loss_type == 'mse':
            custom_loss = F.mse_loss(
                self.rq.vq_layers[codebook_level].embedding.weight,
                expected_codebook_embedding, 
                reduction='mean')
        elif self.custom_loss_type == 'l1':
            aaa=self.rq.vq_layers[codebook_level].embedding.weight
            if len(self.filter_range_cb3rd) == 2:
                aaa = aaa[self.filter_range_cb3rd[0] : self.filter_range_cb3rd[1]]
                if expected_codebook_embedding.shape[0] != self.filter_range_cb3rd[1] -self.filter_range_cb3rd[0]:
                    expected_codebook_embedding = expected_codebook_embedding[self.filter_range_cb3rd[0] : self.filter_range_cb3rd[1]]
            custom_loss = F.l1_loss(
                aaa,
                expected_codebook_embedding, 
                reduction='mean')
        elif self.custom_loss_type == 'cosine':
            cosine_sim = F.cosine_similarity(
                self.rq.vq_layers[codebook_level].embedding.weight, 
                expected_codebook_embedding, 
                dim=1)  # 在特征维度上计算余弦相似度
            custom_loss = 1 - cosine_sim.mean() 
        else:
            raise Exception(f"Do not support this custom_loss_type({self.custom_loss_type})")
            
        return custom_loss
        
    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        
        if self.init_codebook_with_description_embedding:
            if self.need_encoder_for_target_embedding: #如果是从样本中直接计算的话，那还需要按照当时的encoder参数去算一下才能得到codebook的值
                if self.imported_embedding_full_cuda[0] is not None:
                    self.target_codebook_embedding_1st = self.encoder(self.imported_embedding_full_cuda[0].cuda())
                else:
                    assert(len(self.target_corpus_embeddings_from_traindata_1st) > 0)
                    aaa=[]
                    for class_l1 in self.target_corpus_embeddings_from_traindata_1st:
                        tmp = self.encoder(class_l1.cuda())
                        aaa.append(torch.mean(tmp, dim=0))
                    mean_tensors_1 = [ 
                                      torch.mean(self.encoder(class_l1.cuda()), dim=0) 
                                      for class_l1 in self.target_corpus_embeddings_from_traindata_1st 
                        ]
                    self.target_codebook_embedding_1st = torch.stack(mean_tensors_1).cuda()  
                    
                if self.imported_embedding_full_cuda[1] is not None and self.target_codebook_embedding_2nd is None:
                    self.target_codebook_embedding_2nd = self.encoder(self.imported_embedding_full_cuda[1].cuda())
                elif len(self.target_corpus_embeddings_from_traindata_2nd) > 0 and self.target_codebook_embedding_2nd is None:
                    mean_tensors_2 = [ torch.mean(self.encoder(class_l1.cuda()), dim=0) for class_l1 in self.target_corpus_embeddings_from_traindata_2nd ]
                    self.target_codebook_embedding_2nd = torch.stack(mean_tensors_2).cuda()

                if self.imported_embedding_full_cuda[2] is not None and self.target_codebook_embedding_3rd is None:
                    self.target_codebook_embedding_3rd = self.encoder(self.imported_embedding_full_cuda[2].cuda())
                elif len(self.target_corpus_embeddings_from_traindata_3rd) > 0 and self.target_codebook_embedding_3rd is None:
                    mean_tensors_3 = [ torch.mean(self.encoder(class_l1.cuda()), dim=0) for class_l1 in self.target_corpus_embeddings_from_traindata_3rd ]
                    self.target_codebook_embedding_3rd = torch.stack(mean_tensors_3).cuda()

                    
                # if self.imported_embedding_full_cuda[2] != None:
                #     self.imported_embedding_full_cuda[2] = self.encoder(self.imported_embedding_full_cuda[2].cuda())
                    

                first_cb_weights = self.rq.vq_layers[0].embedding.weight
                # 针对第二个码本的目标embedding，判断是否要计算差值（即减去第一个码本embedding）
                if self.custom_codebook_residual == 1 and self.target_codebook_embedding_2nd is not None:
                    
                    code_num = self.target_codebook_embedding_2nd.shape[0]
                    new_tensor = torch.zeros(code_num, self.e_dim)
                    start_idx = 0
                    for i, count in enumerate(self.num_emb_list[1 : (1 + self.num_emb_list[0])]):
                        end_idx = start_idx + count
                        new_tensor[start_idx : end_idx, :] = first_cb_weights[i].unsqueeze(0).expand(count, -1)
                        start_idx = end_idx
            
                    self.target_codebook_embedding_2nd -= new_tensor.detach().cuda()
                
                if self.custom_codebook_residual == 1 and 2 in self.init_class_embedding_codebook_level:
                    cb2_weights = self.rq.vq_layers[1].embedding.weight
                    cb0_code_num = self.rq.vq_layers[0].n_e
                    cb1_code_num = self.rq.vq_layers[1].n_e
                    cb2_code_num = self.rq.vq_layers[2].n_e
                    new_tensor = torch.zeros(cb2_code_num, self.e_dim)
                    start_idx = 0

                    # print(self.num_emb_list)
                    for i in range(cb0_code_num):#0 1 2 3 4
                        for j in range(self.num_emb_list[1 + i]): #j 0 1 2 3 4 5 / 0 1 2 3 
                            count = self.num_emb_list[1 + cb0_code_num + j + sum(self.  num_emb_list[1: 1+i])]
                            end_idx = start_idx + count
                            new_tensor[start_idx : end_idx, :] = cb2_weights[j + sum(self.num_emb_list[1: 1+i])].unsqueeze(0).expand(count, -1) + first_cb_weights[i]
                            start_idx = end_idx

                    self.target_codebook_embedding_3rd -= new_tensor.detach().cuda()
                    

            if self.init_class_embedding_codebook_level == [0]:
                custom_loss = self.custom_loss(0, self.target_codebook_embedding_1st.detach())
            elif self.init_class_embedding_codebook_level == [1]:
                custom_loss = self.custom_loss(1, self.target_codebook_embedding_2nd.detach())
            elif self.init_class_embedding_codebook_level == [0,1]:
                custom_loss = self.custom_loss(0, self.target_codebook_embedding_1st.detach()) + self.custom_loss_weight_gamma * self.custom_loss(1, self.target_codebook_embedding_2nd.detach())
            elif self.init_class_embedding_codebook_level == [2]:
                custom_loss = self.custom_loss(2, self.target_codebook_embedding_3rd)
            elif self.init_class_embedding_codebook_level == [0,1,2]:
                custom_loss = self.custom_loss(0, self.target_codebook_embedding_1st.detach()) + self.custom_loss_weight_gamma * self.custom_loss(1, self.target_codebook_embedding_2nd.detach()) + self.custom_loss_weight_gamma * self.custom_loss(1, self.target_codebook_embedding_2nd.detach())
            else:
                raise Exception(f"Do not support this nit_class_embedding_codebook_level{self.init_class_embedding_codebook_level} right now.")
                
            loss_total += self.custom_loss_weight * custom_loss
            return loss_total, loss_recon, quant_loss, custom_loss

        return loss_total, loss_recon, quant_loss, 0
