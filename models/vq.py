import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm
import numpy as np


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 4,
                 sk_epsilon=0.003, sk_iters=100,):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data, start=0, end=-1):
        if start == 0 and end == self.n_e:
            centers = kmeans(
                data,
                self.n_e,
                self.kmeans_iters,
            )

            self.embedding.weight.data.copy_(centers)
        else:
            if torch.all(self.embedding.weight[start:end]==0).item()==False:
                print(self.embedding.weight[start:end])
            centers = kmeans(
                data,
                end - start,
                self.kmeans_iters,
            )
            self.embedding.weight[start:end].data.copy_(centers)
            if torch.count_nonzero(self.embedding.weight[start:end]).item() != self.e_dim * (end-start):
                print(self.embedding.weight[start:end])
        self.initted = True


    def get_statistics(self, indices, label, size):
        sorted_tensor, idx = torch.unique(label).sort()
        mapping={}
        # results = []
        for i in idx.numpy():
            mapping[i] = sorted_tensor[i].item()
        if size != len(mapping):
            print("mapping=", mapping, " size=", size)
            size = len(mapping)
        for i in range(size):
            index = np.where(label.cpu() == mapping[i])[0]
            unique_elements, counts = np.unique(indices[index].cpu().numpy(), return_counts=True)
            sorted_counts = sorted(zip(unique_elements, counts), key=lambda x: x[1], reverse=True)
            print(sorted_counts)
            # results.append(np.where(indices[index].cpu().numpy() == i)[0])

        # return results

    def vq_init(self, labels, level, x): #x.shape=[12101, 32]是所有样本经过encoder后的向量
        latent = x.view(-1, self.e_dim) #e_dim=32

        # Debug info
        if level < 4:
            # print(self.n_e, latent.shape)
            print("latent=",latent)

        if not self.initted:
            self.init_emb(latent, start=0, end=self.n_e)
            self.initted = True

        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t()) #d is a tensor, d.shape=[12101,256]

        indices = torch.argmin(d, dim=-1)
        x_q = self.embedding(indices).view(x.shape)

        if level == 0:
            print("##################distribution##########\n", torch.unique(indices,return_counts=True))
            self.get_statistics(indices, labels[:,0], self.n_e)

        #print("codebook#######self.embedding.weight=", self.embedding.weight.data)
        return indices, x_q #[12101, 32]
    

    def vq_init_1st_initted_cb(self, labels, x, mapping):
        indices = torch.tensor([mapping[i.item()] for i in labels.flatten()])
        indices = indices.view(labels.shape)

        x_q = self.embedding(indices.to("cuda:0")).view(x.shape)

        return indices, x_q 
    
    #初始化第level级的码本（根节点的level=1）的start-end编号的码
    def vq_init_partial(self, labels, cb_idx, x, start, end, kmeans=False): 
        #x.shape=[12101, 32]即所有样本encoder后的向量
        latent = x.view(-1, self.e_dim) #e_dim=32
        
        if cb_idx == 0:
            label = labels[:, 0]
            sorted_tensor,_ = torch.unique(label).sort()
        elif cb_idx == 1:
            label = labels
            sorted_tensor,_ = torch.unique(label).sort()
        elif cb_idx >= 2:
            assert labels == None
            self.init_emb(x, start, end)
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight[start:end]**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight[start:end].t()) #d is a tensor, d.shape=[12101,256]
            
            indices = torch.argmin(d, dim=-1)
            indices += start
            x_q = self.embedding(indices).view(x.shape)
            
            return indices, x_q
        
        # assert(len(sorted_tensor) == end-start)
        #如果之前初始化过就不再初始化了
        if not self.initted:
            for i in range(0, end - start):
                indices = (label == sorted_tensor[i]).nonzero(as_tuple=True)[0]
                selected_embeddings=latent[indices]
                mean_embedding = torch.mean(selected_embeddings, dim=0)
                self.embedding.weight[start + i].data.copy_(mean_embedding)
        
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight[start:end]**2, dim=1, keepdim=True).t()- \
                    2 * torch.matmul(latent, self.embedding.weight[start:end].t()) #d is a tensor, d.shape=[12101,256]
            
        indices = torch.argmin(d, dim=-1)
        self.get_statistics(indices + start, label, end-start)
    
        sorted_dict = {value: index for index, value in enumerate(sorted_tensor.numpy())}
        indices = start + torch.tensor([sorted_dict[value] for value in label.numpy()])

            
        x_q = self.embedding(indices.to("cuda:0")).view(x.shape)

        return indices, x_q #[12101, 32]
    
    
    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def get_distance(self, x, need_inference_first_class=False, labels_mapping={}):
        latent = x.view(-1, self.e_dim)
        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if not need_inference_first_class:
            return d
        else:
            tmp_first_cb_embedding=[]
            for cls in labels_mapping['level-0-mapping'].keys():
                
                start = list(labels_mapping[cls].values())[0]
                end = list(labels_mapping[cls].values())[-1] + 1
                print(cls, labels_mapping[cls], start, end)
                
                embedding = torch.mean(self.embedding.weight[start:end], dim=0)
                tmp_first_cb_embedding.append(embedding)
            first_cb_embedding = torch.stack(tmp_first_cb_embedding, dim=0)
            fisrt_cb_distance = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(first_cb_embedding**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, first_cb_embedding.t())
            return torch.cat([fisrt_cb_distance, d], dim=1)
            # d = fisrt_cb_distance + d
        return d
            
    def forward(self, x, start, end, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        # if not self.initted and self.training:
        #     self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight[start:end]**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight[start:end].t())
            

        
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
            indices += start
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
            indices += start

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


