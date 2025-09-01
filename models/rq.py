import torch
import torch.nn as nn
import numpy as np

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,
                 architecture = '', num_quantizers=3, last_codebook_shared=False,
                 labels_mapping={},
                 init_codebook_with_description_embedding=False,
                 init_class_embedding_codebook_level = -1,
                 codebook_shared_status = [False, False, True, True, True]):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = num_quantizers
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.architecture = architecture
        self.last_codebook_shared = last_codebook_shared
        self.labels_mapping = labels_mapping
        self.init_codebook_with_description_embedding = init_codebook_with_description_embedding,
        self.init_class_embedding_codebook_level = init_class_embedding_codebook_level
        self.codebook_shared_status = codebook_shared_status
        
        if self.architecture == 'normal':
            self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])
        elif self.architecture == "tree_residual_uniform":
            # 定义树状结构的码本
            tree_module = []
            tmp_list = n_e_list
            if len(tmp_list) > 10:
                self.codebook_shared_status = [False, False, False, True, True]
            quantizer_idx = 0
            current_quantier_codenum = 0
            while(len(tmp_list) > 0):
                if quantizer_idx == 0: #如果是第一个码本，则个数=n_e_list[0]
                    cb_list = n_e_list[0: 1]
                    tmp_list = tmp_list[1:]
                    current_quantier_codenum =  sum(cb_list)
                else:
                    if not self.codebook_shared_status[quantizer_idx]: #当前码本是树状结构
                        cb_list = tmp_list[0: current_quantier_codenum]
                        tmp_list = tmp_list[current_quantier_codenum:]
                        current_quantier_codenum =  sum(cb_list)
                    else:
                        current_quantier_codenum = tmp_list[0]
                        tmp_list = tmp_list[1:]
                tree_module.append(VectorQuantizer(current_quantier_codenum,
                                                e_dim, 
                                                beta=self.beta,
                                                kmeans_init = self.kmeans_init,
                                                kmeans_iters = self.kmeans_iters,
                                                sk_epsilon=0.0
                                                ))
                quantizer_idx += 1
            self.vq_layers = nn.ModuleList(tree_module)
            if len(tree_module) != self.num_quantizers:
                raise Exception(f"model vq_layers({len(self.vq_layers)}) != num_quantizers({num_quantizers})")

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def vq_normal_init(self, labels, x):
        print("--------->rq2.py vq_normal_init 随机采样10000个")
        x_q = 0
        
        if x.shape[0] > 10000:
            num_samples = 5000
            # 无放回抽样
            indices = torch.randperm(x.size(0))[:num_samples]
            residual = x[indices]
            labels = labels[indices]
        else:
            residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            _, x_res = quantizer.vq_init(labels, idx, residual)
            residual = residual - x_res
            x_q = x_q + x_res

    def get_codebook_start_end(self, code_idx_list):
        # import ipdb
        # ipdb.set_trace()
        cb_idx = len(code_idx_list) - 1
        quantizer = self.vq_layers[cb_idx]
        start = 0
        end = quantizer.n_e
        
        #cb_idx=0表示根节点
        #cb_idx=1表示根节点的子节点（这个肯定是树状结构的）
        # cb_idx=2表示根节点的孙子节点（结构是否为树状根据codebook_shared_status[2]判断， True为共享即非树状，False为树状结构）
        #start和end表示当前码本里的第start到end编号的码
        #例如code_idx_list=[0,0]表示根节点码本的下标=0即第一个码的子节点
        #                  [0,1]表示根节点码本的下标=1即第二个码的子节点，所以他的start=第一个分类的二级分类个数
        #                  [0,2]表示根节点码本的下标=2即第三个码的子节点，所以他的start=第一个分类和第二个分子的二级分类个数之和
        if cb_idx == 1: #idx=1的码本肯定是分开的
            start = sum(self.n_e_list[1 : 1 + code_idx_list[1]])
            end = start + self.n_e_list[1 + code_idx_list[1]]
        elif cb_idx == 2:
            if not self.codebook_shared_status[cb_idx]:
                #这一级码本开始的下标
                this_level_nelist_start_idx = 1 + self.n_e_list[0]
                this_level_code_num = sum(self.n_e_list[1 : this_level_nelist_start_idx])
                tmp_n_e_list = self.n_e_list[this_level_nelist_start_idx : this_level_nelist_start_idx + this_level_code_num ]
                
                before =  sum(self.n_e_list[1 : 1+code_idx_list[1]])
                count = tmp_n_e_list[before + code_idx_list[2]]
                start = sum(tmp_n_e_list[0 : before + code_idx_list[2]])
                end = start + count
        
        return start, end
    
    def residual_tree_init(self, labels, x, code_idx_list=[0], is_initted=False):
        #第一个码本是从checkpoint里继承来了，只需要初始化后面的码本即可。
        # if code_idx_list == [0,0,0,0]:
        #     import ipdb
        #     ipdb.set_trace()
        all_indices = []
        x_q = 0
        residual = x
        cb_idx = len(code_idx_list) - 1
        quantizer = self.vq_layers[cb_idx]
        tabs = "\t\t" * cb_idx
        
        start, end = self.get_codebook_start_end(code_idx_list)
        print(f"{tabs}------>residual_tree_init  {x.shape} {code_idx_list}  start-end= {start}-{end}")

        if is_initted:
            assert(code_idx_list == [0] and quantizer.initted)
            indices, x_res = quantizer.vq_init_1st_initted_cb(labels[:,0], x, self.labels_mapping['level-0-mapping'])
        else:
            indices, x_res = quantizer.vq_init_partial(labels, cb_idx, residual, start, end)

        x_q = x_q + x_res
        residual = x - x_res
        all_indices.append(indices.cpu())
        
        #如果是最后一个码本，则直接返回
        if cb_idx == self.num_quantizers - 1:
            all_indices = torch.stack(all_indices, dim=-1)
            print(f"{tabs}<------residual_tree_init  {x.shape}  {code_idx_list}")
            return all_indices, x_q

        #处理下一级码本：
        if not self.codebook_shared_status[cb_idx + 1]:
            # 如果下一级的码本是非共享的，即需要按照树状结构执行，则递归地去处理该码本对应的子码本
            if cb_idx < self.num_quantizers - 2 and not self.codebook_shared_status[cb_idx + 2]:
                new_indices = torch.zeros(x.shape[0], 2, dtype=torch.int)
            else:
                new_indices = torch.zeros(x.shape[0], 1, dtype=torch.int)
                
            for i in range(end - start):
                #顺序遍历当前码本的码对应的子节点
                index = np.where(indices.cpu() == i + start)[0]
                
                tmp_list = code_idx_list.copy()
                tmp_list.append(i)
                if len(index) == 0:
                    print("Note:no item in ",tmp_list, "  x.shape=",x.shape)
                    continue
                
                sub_sets_indices =  None
                if cb_idx == 0:
                    sub_sets_indices = labels[index][:, 1]
                    
                sub_indices, sub_x_res = self.residual_tree_init(sub_sets_indices, residual[index], tmp_list)

                for in_i, idx in enumerate(index):
                    x_q[idx] += sub_x_res[in_i]
                    if (new_indices.shape[-1] == 1 and new_indices[idx].item() == 0) or (new_indices.shape[-1] == 2 and new_indices[idx][0].item() == 0 and new_indices[idx][1].item() == 0):
                        new_indices[idx] = sub_indices[in_i]
                    else:
                        raise Exception("index conflict.")

            if end != quantizer.n_e:
                if cb_idx == 1:#[0,0], [0, 1]进入到这里 
                    assert(new_indices.shape[1] == 1) 
                    all_indices.append(new_indices.cpu().squeeze(1))
                    all_indices = torch.stack(all_indices, dim=-1)
                    print(f"{tabs}<------residual_tree_init  {x.shape}  {code_idx_list}")
                    return all_indices, x_q
                else:
                    import ipdb
                    ipdb.set_trace()
                    raise Exception("TODO")
            else:
                if cb_idx == 0:
                    tmp_list = [0, 0, 0]
                    if new_indices.shape[1] == 1:
                        all_indices.append(new_indices.cpu().squeeze(1))
                    elif new_indices.shape[1] == 2:
                        tmp_list = [0, 0, 0, 0]
                        tensor1, tensor2 = torch.unbind(new_indices, dim=1)
                        all_indices.append(tensor1)
                        all_indices.append(tensor2)
                    elif new_indices.shape[1] > 2:
                        import ipdb
                        ipdb.set_trace()
                        raise Exception("TODO")
                    
                    new_indices2, x_res2 = self.residual_tree_init(None,  x - x_q, tmp_list)
                    x_q += x_res2
                    if new_indices2.shape[1] == 1:
                        all_indices.append(new_indices2.cpu().squeeze(1))
                    elif new_indices2.shape[1] == 2:
                        tensor1, tensor2 = torch.unbind(new_indices2, dim=1)
                        all_indices.append(tensor1)
                        all_indices.append(tensor2)
                    elif new_indices2.shape[1] > 2:
                        import ipdb
                        ipdb.set_trace()
                        raise Exception("TODO")
                else:
                    import ipdb
                    ipdb.set_trace()
                    raise Exception("TODO")
        
        #如果下一级码本是共享的，则等到所有本级码本都初始化完再初始化下一级码本。
        elif self.codebook_shared_status[cb_idx] :  #如果当前这个码本就是共享的，那下一个码本肯定也是共享的，递归处理
            tmp_list = [0] * (cb_idx + 2)
            indices, x_res = self.residual_tree_init(None,  x - x_q, tmp_list)
            x_q += x_res
            all_indices.append(indices.cpu().squeeze(1))

        
        all_indices = torch.stack(all_indices, dim=-1)
        print(f"{tabs}<------residual_tree_init  {x.shape}  {code_idx_list}")
    
        return all_indices, x_q
    
    
    #和init是同样的原理
    def residual_tree_quantizer(self, x, code_idx_list=[0]):
        all_losses = []
        all_indices = []
        cb_idx = len(code_idx_list) - 1
        quantizer = self.vq_layers[cb_idx]
        x_q = 0
        residual = x
        
        start, end = self.get_codebook_start_end(code_idx_list)
        x_res, loss, indices = quantizer(residual, start, end)
        
        if  torch.isnan(loss):
            ipdb.set_trace()

        all_losses.append(loss)
        x_q = x_q + x_res
        residual = x - x_res
        all_indices.append(indices)

        if cb_idx < self.num_quantizers - 1:
            if not self.codebook_shared_status[cb_idx + 1]:
                # 如果下一级的码本是非共享的，即需要按照树状结构执行，则递归地去处理该码本对应的子码本
                if cb_idx < self.num_quantizers - 2 and not self.codebook_shared_status[cb_idx + 2]:
                    new_indices = torch.full((x.shape[0], 2), -1, dtype=torch.int).cuda()
                else:
                    new_indices = torch.full((x.shape[0], 1), -1, dtype=torch.int).cuda()
                    
                for i in range(end - start):
                    index = np.where(indices.cpu() == i + start)[0]
                    
                    tmp_list = code_idx_list.copy()
                    tmp_list.append(i)
                    if len(index) == 0:
                        continue

                    sub_x_res, loss, sub_indices = self.residual_tree_quantizer(residual[index], tmp_list)
                    all_losses.append(loss)
                    x_q[index] += sub_x_res
                    new_indices[index] = sub_indices.to(new_indices.dtype)
                    
                    # for in_i, idx in enumerate(index):
                    #     x_q[idx] += sub_x_res[in_i]
                    #     new_indices[idx] = sub_indices[in_i]

                    
                if new_indices.shape[1] == 1:
                    all_indices.append(new_indices.to("cuda:0").squeeze(1))
                elif new_indices.shape[1] == 2:
                    tmp_list = [0, 0, 0, 0]
                    tensor1, tensor2 = torch.unbind(new_indices, dim=1)
                    all_indices.append(tensor1.to("cuda:0"))
                    all_indices.append(tensor2.to("cuda:0"))

            if cb_idx == 0 or self.codebook_shared_status[cb_idx]:
                if cb_idx == 0:
                    tmp_list = [0,0,0] if new_indices.shape[1] == 1 else [0,0,0,0]
                else:
                    tmp_list = (cb_idx + 2) * [0]
                sub_x_res, loss, sub_indices = self.residual_tree_quantizer(x - x_q, code_idx_list=tmp_list)
                x_q += sub_x_res
                if sub_indices.shape[1] == 1:
                    all_indices.append(sub_indices.to("cuda:0").squeeze(1))
                else:
                    tensor1, tensor2 = torch.unbind(sub_indices, dim=1)
                    all_indices.append(tensor1.to("cuda:0"))
                    all_indices.append(tensor2.to("cuda:0"))
                all_losses.append(loss)
                
        mean_losses = torch.stack(all_losses).mean()
        if  torch.isnan(mean_losses):
            ipdb.set_trace()
        all_indices = torch.stack(all_indices, dim=-1)

        return  x_q, mean_losses, all_indices
    
    def get_distance(self, x, codebook_id=0, need_inference_first_class=False, labels_mapping={}):
        residual = x
        results = []
        if self.architecture == "normal":
            for idx, quantizer in enumerate(self.vq_layers):
                predict_distance = quantizer.get_distance(residual, need_inference_first_class, labels_mapping)
                results.append(predict_distance)
                x_res, _, _ = quantizer(residual, start=0, end=quantizer.n_e)
                residual = residual - x_res
                if idx == codebook_id:
                    break
        else:
            for idx, quantizer in enumerate(self.vq_layers):
                predict_distance = quantizer.get_distance(residual, need_inference_first_class, labels_mapping)
                results.append(predict_distance)
                x_res, _, _ = quantizer(residual, start=0, end=quantizer.n_e)
                residual = residual - x_res
                if idx == codebook_id:
                    break
        
        return results  

                    
    def forward(self, x, use_sk=True):
        if self.architecture == "normal":
            all_losses = []
            all_indices = []

            x_q = 0
            residual = x
            
            for quantizer in self.vq_layers:
                x_res, loss, indices = quantizer(residual, start=0, end=quantizer.n_e, use_sk=use_sk)

                residual = residual - x_res
                x_q = x_q + x_res

                all_losses.append(loss)
                all_indices.append(indices)

            mean_losses = torch.stack(all_losses).mean()
            all_indices = torch.stack(all_indices, dim=-1)
        else:
            x_q, mean_losses, all_indices = self.residual_tree_quantizer(x, [0])

        return x_q, mean_losses, all_indices
