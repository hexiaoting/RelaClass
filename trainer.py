import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
import json
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir,set_color,get_local_time,delete_file
import os
from sklearn.metrics import classification_report
import heapq
from sklearn.metrics import f1_score


class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_precision = -np.inf
        self.best_secondlevel_precision = -np.inf
        self.best_thirdlevel_precision = -np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_precision_ckpt = "best_precision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx, column_name):
        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        train_quant_loss = 0
        total_custom_loss = 0

        # for _, data in enumerate(iter_data):
        for _, data in enumerate(train_data):
            if isinstance(data, dict):
                embedding = data[column_name].to(self.device)
            else:
                embedding = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, _ = self.model(embedding)
            loss, loss_recon, loss_quant, loss_custom = self.model.compute_loss(out, rq_loss, xs=embedding)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            # print(self.scheduler.get_last_lr())
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            train_quant_loss += loss_quant.item()
            total_custom_loss += loss_custom

        return total_loss, total_recon_loss, train_quant_loss, total_custom_loss

    #将索引转为正确的码本编号，打印出来看看
        #    比如以下结果表明有4类样本：
        #         第0类样本和其他样本区别度较大，84%都识别正确了，且映射到码本的第一个码
        #         第1类样本有一大半映射到了第4个码本
        #         但是第2和3类样本没有区分开来，都映射到了第3个码，最好是分别映射到第2、3个码，不要重。
        #       [(0, 8430), (1, 623), (3, 593), (2, 354)]
        #       [(3, 7096), (2, 2000), (1, 851), (0, 53)]
        #       [(3, 5240), (1, 2560), (2, 1791), (0, 409)]
        #        ****************Error************
        #       [(1, 5523), (2, 3691), (3, 693), (0, 93)]
    def convert_true_lable_to_cbindex(self, indices, labels, level=0, debug=False, parent_cb_id = -1):
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

        #S2. 分别打印每一个类别通过码本获得的索引编号，比如看看beauty这个一类别在码本中分别被映射到哪个码本，每个码本映射的数量是多少
        #    最佳的结果是所有beauty的样本的码本索引都一样，toys类别的样本的码本indices都一样（但和beauty是不一样的），这才是最佳结果
        new_label = np.zeros(label.shape[0], dtype=np.int64)
        # used_codes = set()
        if level == 0:
            print("codebook-l0-result:")
        distinct_class_num_1st = 0
        distinct_class_num_2nd = 0
        distinct_class_num_3rd = 0
        correct_samples_num_1st = {}
        correct_samples_num_2nd = 0
        correct_samples_num_3rd = 0
        codebook_order_1st = ""
        codebook_order_2nd = ""
        codebook_order_3rd = ""
        for i in range(class_num):
            if level == 2 and self.model.rq.codebook_shared_status[2]:
                break
            index = np.where(label == lableid2idx_mapping[i])[0]
            unique_elements, counts = np.unique(indices[:,level][index], return_counts=True)
            sorted_counts = sorted(zip(unique_elements, counts), key=lambda x: x[1], reverse=True)
            # 用以下方法可以打印某些二级分类情况
            # print("\t" * level, sorted_counts[:6])
            # if level == 0 and (i == 2 or i == 3):
            start_tmp = 0
            if level == 1:
                start_tmp = sum(self.model.num_emb_list[1: 1+ parent_cb_id])
            elif level == 2:
                cb3rd_start = self.model.num_emb_list[0] + 1
                start_tmp = sum(self.model.num_emb_list[cb3rd_start: cb3rd_start+ parent_cb_id])
            if level == 0 and len(self.args.filter_range_cb3rd) ==2:
                if self.args.filter_range_cb3rd[0] == 0:
                    start_tmp = 0
                elif self.args.filter_range_cb3rd[0] == 32:
                    start_tmp = 1
                elif self.args.filter_range_cb3rd[0] == 70:
                    start_tmp = 2
                elif self.args.filter_range_cb3rd[0] == 97:
                    start_tmp = 3
                elif self.args.filter_range_cb3rd[0] == 211:
                    start_tmp = 4
                else:
                    raise Exception(" self.args.filter_range_cb3rd wrong!!!")
            sub_index = index[np.where(indices[:,level][index] == (start_tmp + i))]
            if level == 0 and self.args.init_class_embedding_codebook_level != [0]:
                 #在这个一级码本 i 下正确的分类的样本下标，看看他们的二级分类效果如何
                # _,correct_sample_num_2nd,_,distinct_2nd,cb_2nd_tmp,_,_ = self.convert_true_lable_to_cbindex(indices[sub_index], 
                codebook_2nd_res = self.convert_true_lable_to_cbindex(indices[sub_index], 
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
                codebook_3rd_res = self.convert_true_lable_to_cbindex(indices[sub_index], 
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
            if sorted_counts[0][0]==start_tmp + i:
                distinct_class_num_1st += 1

        # if level == 1:
        #     print(f"\t\tdistincted_class_num={len(distincted_samples_num.keys())}, distincted_samples_num={sum(distincted_samples_num.values())}")
        res = {}
        res['new_label'] = new_label
        res['distinct_class_num'] = [distinct_class_num_1st, distinct_class_num_2nd, distinct_class_num_3rd]
        res['correct_sample_num'] = [sum(correct_samples_num_1st.values()), correct_samples_num_2nd, correct_samples_num_3rd]
        res['codebook_order'] = [codebook_order_1st, codebook_order_2nd, codebook_order_3rd]
        return res
        # return new_label, sum(distincted_samples_num_1st.values()), sub_level_correct_sample_num,  \
        #         len(distincted_samples_num_1st.keys()), codebook_order_1st,\
        #        distincted_samples_num_2nd, codebook_order_2nd

    def print_satistics(self, all_indices, all_labels):
        stats = {}
        label_keys = []
        index_values = []
        all_labels = [[str(x) for x in row] for row in all_labels.tolist()]
        all_indices = [[str(x) for x in row] for row in all_indices.tolist()]
        for i, label in enumerate(all_labels):
            label_key = '-'.join(label)
            index_value = '-'.join(all_indices[i][:2])
            if label_key not in stats:
                stats[label_key] = {}
            if index_value not in stats[label_key].keys():
                stats[label_key][index_value] = 0
            stats[label_key][index_value] += 1
            if label_key not in label_keys:
                label_keys.append(label_key)
            if index_value not in index_values:
                index_values.append(index_value)
        
        label_keys.sort()
        index_values.sort()
        
        for v in index_values:
            print(f"\t\t{v}", end='')
        print()
        for k in label_keys:
            label_count = sum(1 for row in all_labels if '-'.join(map(str, row)) == k)
            # if k[:3] != '0-1':
            #     continue
            print(f"{k}({label_count})", end='\t')
            for index_column in index_values:
                if index_column not in stats[k].keys():
                    print(0, end='\t\t')
                else:
                    print(stats[k][index_column], end='\t\t')
            print()
        
    def get_codebook_mapping(self, labels_mapping=None, level = 0):
        if labels_mapping is None:
            return mapping
        mapping = labels_mapping['level-0-mapping']
        res = {}
        count = 0
        for k_0 in mapping.keys():
            if level == 0:
                res[k_0] = count
                count += 1
            else:
                for k_1 in labels_mapping[k_0].keys():
                    if level == 1:
                        res[k_1] = count
                        count += 1
                    else:
                        for k_2 in labels_mapping[k_1].keys():
                            if level == 2:
                                if len(self.args.filter_range_cb3rd) == 2:
                                    res[k_2] = count + self.args.filter_range_cb3rd[0]
                                else:
                                    res[k_2] = count
                                count += 1
        return res



    @torch.no_grad()
    def _valid_epoch(self, valid_data, epoch_idx, column_name="embedding", labels_mapping=None):
        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        #hwt 修改
        all_indices = []
        all_labels = []
        for batch_idx, data in enumerate(iter_data):
            if isinstance(data, dict):
                embedding = data[column_name].to(self.device)
                labels = data['labels']
            else:
                embedding = data.to(self.device)
                labels = torch.empty(0)
            num_sample += len(embedding)
            indices = self.model.get_indices(embedding)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

            for v in indices:
                all_indices.append(v)
            for v in labels.numpy():
                all_labels.append(v)

        all_indices = np.vstack(all_indices)
        all_labels = np.vstack(all_labels)

        def sparse_to_binary(raw_labels, num_classes):
            #将稀疏的类别路径列表转换为多标签二值矩阵
            n_samples = len(raw_labels)
            binary_matrix = np.zeros((n_samples, num_classes), dtype=int)

            for i, labels in enumerate(raw_labels):
                binary_matrix[i, labels] = 1  # 在指定索引处设为 1

            return binary_matrix

        if 2 in self.args.init_class_embedding_codebook_level:
            if max(max(labels) for labels in all_labels) + 1 != 322:
                gold_labels = []
                predict_labels = []
                for i, label in enumerate(all_labels):
                    gold_label = [
                        labels_mapping['level-0-mapping'][label[0]],
                        labels_mapping[label[0]][label[1]] + 5,
                        labels_mapping[label[1]][label[2]] + 57
                    ]
                    predict_label = [
                        all_indices[i][0],
                        all_indices[i][1] + 5,
                        all_indices[i][2] + 57
                    ]
                    gold_labels.append(gold_label)
                    predict_labels.append(predict_label)
            num_classes = max(max(labels) for labels in gold_labels) + 1
            
            micro_f1 = f1_score(sparse_to_binary(gold_labels, num_classes), sparse_to_binary(predict_labels, num_classes), average='micro')
            macro_f1 = f1_score(sparse_to_binary(gold_labels, num_classes), sparse_to_binary(predict_labels, num_classes), average='macro')
            tmp_count = 0
            for pred, gt in zip(predict_labels, gold_labels):
                if set(gt) == set(pred):
                    tmp_count+=1
            precision_3 = tmp_count / len(gold_labels)
            
            print(f"Micro-F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}, precision@3:{precision_3:.4f}")

        res_dict = self.convert_true_lable_to_cbindex(all_indices, all_labels, level=0)

        # self.print_satistics(all_indices, all_labels)

        # if precision > 0.8:
        #     print(classification_report(label_true, all_indices[:,0], target_names=["grocery_gourmet_food","toys_games","beauty","health_personal_care","baby_products","pet_supplies"]))

        # return precision, corrent_num_2nd/num_sample, distincted_class_num_1st, codebook_order_1st, distincted_class_num_2nd, codebook_order_2nd
        return res_dict, num_sample

    def _save_checkpoint(self, epoch, precision=1, secondlevel_precision=1,  thirdlevel_precision=1, ckpt_file=None):
        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_precision_%.4f_%.4f_%.4f_model.pth' % (epoch, precision, secondlevel_precision, thirdlevel_precision))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_precision": self.best_precision,
            "best_secondlevel_precision":self.best_secondlevel_precision,
            "best_thirdlevel_precision":self.best_thirdlevel_precision,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, quant_loss, custom_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.8f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.8f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("quantification loss", "blue") + ": %.8f" % quant_loss
        train_loss_output +=", "
        train_loss_output += set_color("custom loss", "blue") + ": %.8f" % custom_loss
        return train_loss_output + "]"


    def fit(self, data, test_data, column_name, labels_mapping):
        #第一步：初始化模型参数
        if self.args.init_method in ["load_from_normalrqvae_ckpt"]: #从init_state文件里初始化
            if len(self.args.codebook1_order) == 0:
                raise Exception("You should set the first codebook index order, like for Amazon-531 dataset, noraml RQVAE's first codebook is 0 2 1 5 4 3")
            
            if self.args.ckpt_path is None:
                self.args.ckpt_path = f"{self.args.ckpt_dir}/init_state"
            state = torch.load(self.args.ckpt_path)
            
            pretrained_state_dict = state['state_dict']
            model_state_dict = self.model.state_dict()

            # 仅保留预训练模型中与当前模型键名相同的项
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
            
            vq_layer_0_name = 'rq.vq_layers.0.embedding.weight'
            if vq_layer_0_name not in pretrained_state_dict.keys():
                raise Exception(f"{vq_layer_0_name} is not the same in ckpt_path model and this tree model, Please check it")
            vq_layer_0_embedding = pretrained_state_dict.pop(vq_layer_0_name)
            for i, true_idx in enumerate(self.args.codebook1_order):
                model_state_dict[vq_layer_0_name][i] = vq_layer_0_embedding[true_idx]
            self.model.rq.vq_layers[0].initted = True
            
            
            # 更新当前模型的状态字典
            model_state_dict.update(pretrained_state_dict)
            # 加载更新后的状态字典到模型中
            self.model.load_state_dict(model_state_dict)
            
            # for layer in self.model.rq.vq_layers:
            #     layer.initted = True
            for param in self.model.encoder.mlp_layers.parameters():
                param.requires_grad = False
            for n, p in self.model.named_parameters():
                if n in pretrained_state_dict.keys():
                    p.requires_grad = False
                elif n == vq_layer_0_name:
                    print("p.requires_grad = Falsep.requires_grad = Falsep.requires_grad = Falsep.requires_grad = Falsep.requires_grad = Falsep.requires_grad = Falsep.requires_grad = False")
                    p.requires_grad = False
        
            self.model.eval()
            self.model.vq_initialization(
                data.dataset['labels'], 
                data.dataset[column_name].to(self.device), 
                first_cb_is_initted = True)
            self._save_checkpoint(epoch=0, ckpt_file="init_state_tree")
        elif self.args.init_method in ["load_from_ckpt"]:
            if self.args.ckpt_path is None:
                self.args.ckpt_path = f"{self.args.ckpt_dir}/init_state_tree"
            state = torch.load(self.args.ckpt_path)
            if self.args.codebook1_order is not None and len(self.args.codebook1_order) == self.args.num_emb_list[0]:
                pretrained_state_dict = state['state_dict']
                model_state_dict = self.model.state_dict()
                pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
                vq_layer_0_name = 'rq.vq_layers.0.embedding.weight'
                if vq_layer_0_name not in pretrained_state_dict.keys():
                    raise Exception(f"{vq_layer_0_name} is not the same in ckpt_path model and this tree model, Please check it")
                vq_layer_0_embedding = pretrained_state_dict.pop(vq_layer_0_name)
                for i, true_idx in enumerate(self.args.codebook1_order):
                    model_state_dict[vq_layer_0_name][i] = vq_layer_0_embedding[true_idx]
                # 更新当前模型的状态字典
                model_state_dict.update(pretrained_state_dict)
                # 加载更新后的状态字典到模型中
                self.model.load_state_dict(model_state_dict)
            else:
                self.model.load_state_dict(state['state_dict'])
            for layer in self.model.rq.vq_layers:
                layer.initted = True
                
            # if 0 not in self.args.init_class_embedding_codebook_level:
            #     self.model.rq.vq_layers[0].requires_grad = False
            #     for param in self.model.encoder.mlp_layers.parameters():
            #         param.requires_grad = False
            #     for param in self.model.rq.vq_layers[0].parameters():
            #         param.requires_grad = False
            
            
            #     if self.args.init_class_embedding_codebook_level == [2]:
            #         # self.model.init_target_cb_embeddding(data, labels_mapping, self.args.column_name, self.ckpt_dir, using_upload_file = False, process_cb_idx=[2])
            #         self.model.rq.vq_layers[1].requires_grad = False
            #         for param in self.model.rq.vq_layers[1].parameters():
            #             param.requires_grad = False
            
            self.model.eval()
            if self.args.target_codebook_generate_method == "system":
                self.model.init_target_cb_embeddding(data, labels_mapping, self.args.column_name, self.ckpt_dir, using_upload_file = False, process_cb_idx=self.args.init_class_embedding_codebook_level, target_datasets_path=self.args.target_codebook_system_datasets)
            elif self.args.target_codebook_generate_method == "user":
                self.model.init_target_cb_embeddding(None, None, None, None,  using_upload_file = True, process_cb_idx=self.args.init_class_embedding_codebook_level)
            
        elif self.args.init_method in ["load_from_ckpt-but_reinit_cb234_using_label", "load_from_ckpt-but_reinit_cb234_using_kmeans"]:
            if self.args.ckpt_path is None:
                self.args.ckpt_path = f"{self.args.ckpt_dir}/init_state_tree"
            state = torch.load(self.args.ckpt_path)
            
            pretrained_state_dict = state['state_dict']
            model_state_dict = self.model.state_dict()
            keys_to_remove = []
            for key in pretrained_state_dict.keys():
                # 筛选出以 'encoder.' 开头的键 或者 键为 'rq.vq_layers.0.embedding.weight'
                if not key.startswith('encoder.') and key != 'rq.vq_layers.0.embedding.weight':
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                pretrained_state_dict.pop(key)
            
            # 更新当前模型的状态字典
            if self.args.codebook1_order is not None and len(self.args.codebook1_order) == self.args.num_emb_list[0]:
                vq_layer_0_name = 'rq.vq_layers.0.embedding.weight'
                if vq_layer_0_name not in pretrained_state_dict.keys():
                    raise Exception(f"{vq_layer_0_name} is not the same in ckpt_path model and this tree model, Please check it")
                vq_layer_0_embedding = pretrained_state_dict.pop(vq_layer_0_name)
                for i, true_idx in enumerate(self.args.codebook1_order):
                    model_state_dict[vq_layer_0_name][i] = vq_layer_0_embedding[true_idx]

            model_state_dict.update(pretrained_state_dict)
            self.model.load_state_dict(model_state_dict)
            
            self.model.rq.vq_layers[0].requires_grad = False
            for param in self.model.encoder.mlp_layers.parameters():
                param.requires_grad = False
            for param in self.model.rq.vq_layers[0].parameters():
                param.requires_grad = False
                
            self.model.rq.vq_layers[0].initted = True
            
            
            self.model.eval()
            if self.args.target_codebook_generate_method == "system":
                self.model.init_target_cb_embeddding(data, labels_mapping, self.args.column_name, self.ckpt_dir, using_upload_file = False, process_cb_idx=[1], target_datasets_path=self.args.target_codebook_system_datasets)
            elif self.args.target_codebook_generate_method == "user":
                self.model.init_target_cb_embeddding(None, None, None, None,  using_upload_file = True, process_cb_idx=[1])
            else:
                raise Exception(f"Wrong target_codebook_generate_method {self.args.target_codebook_generate_method}")
                
            using_kmeans = False
            if self.args.init_method == "load_from_ckpt-but_reinit_cb234_using_kmeans":
                using_kmeans = True
            self.model.vq_initialization(
                data.dataset['labels'], 
                data.dataset[column_name].to(self.device), 
                first_cb_is_initted = True,
                using_kmeans = using_kmeans)
            self._save_checkpoint(epoch=0, ckpt_file="init_state_tree")
        elif self.args.init_method in ["load_from_ckpt-but_reinit_cb34_using_label", "load_from_ckpt-but_reinit_cb34_using_kmeans"]:
            if self.args.ckpt_path is None:
                self.args.ckpt_path = f"{self.args.ckpt_dir}/init_state_tree"
            state = torch.load(self.args.ckpt_path)
            
            pretrained_state_dict = state['state_dict']
            model_state_dict = self.model.state_dict()
            keys_to_remove = []
            for key in pretrained_state_dict.keys():
                # 筛选出以 'encoder.' 开头的键 或者 键为 'rq.vq_layers.0.embedding.weight'
                if not key.startswith('encoder.') and key != 'rq.vq_layers.0.embedding.weight' and key !=  'rq.vq_layers.1.embedding.weight':
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                pretrained_state_dict.pop(key)
            
            # 更新当前模型的状态字典
            model_state_dict.update(pretrained_state_dict)
            self.model.load_state_dict(model_state_dict)
            
            self.model.rq.vq_layers[0].requires_grad = False
            self.model.rq.vq_layers[1].requires_grad = False
            for param in self.model.encoder.mlp_layers.parameters():
                param.requires_grad = False
            for param in self.model.rq.vq_layers[0].parameters():
                param.requires_grad = False
            for param in self.model.rq.vq_layers[1].parameters():
                param.requires_grad = False
                
            self.model.rq.vq_layers[0].initted = True
            self.model.rq.vq_layers[1].initted = True
            
            
            self.model.eval()
            if self.args.target_codebook_generate_method == "system":
                self.model.init_target_cb_embeddding(data, labels_mapping, self.args.column_name, self.ckpt_dir, using_upload_file = False, process_cb_idx=[2], target_datasets_path=self.args.target_codebook_system_datasets)
            elif self.args.target_codebook_generate_method == "user":
                self.model.init_target_cb_embeddding(None, None, None, None,  using_upload_file = True, process_cb_idx=[2])
            #else:
            #    raise Exception(f"Wrong target_codebook_generate_method {self.args.target_codebook_generate_method}")
                
            using_kmeans = False
            if self.args.init_method == "load_from_ckpt-but_reinit_cb34_using_kmeans":
                using_kmeans = True
            self.model.vq_initialization(
                data.dataset['labels'], 
                data.dataset[column_name].to(self.device), 
                first_cb_is_initted = True,
                using_kmeans = using_kmeans)
            self._save_checkpoint(epoch=0, ckpt_file="init_state_tree")

        elif self.args.init_method in ["full_init"]:
            self.model.eval()
            self.model.vq_initialization(
                data.dataset['labels'], 
                data.dataset[column_name].to(self.device), 
                first_cb_is_initted = False)
            if self.args.target_codebook_generate_method == "system":
                self.model.init_target_cb_embeddding(data, labels_mapping, self.args.column_name, self.ckpt_dir, using_upload_file = False, 
                                                     process_cb_idx=self.args.init_class_embedding_codebook_level, 
                                                     target_datasets_path=self.args.target_codebook_system_datasets,
                                                     remove_first_label=((self.args.dataset!= "wos" and self.args.filter_dataset_first_id != -1)))
            elif self.args.target_codebook_generate_method == "user":
                self.model.init_target_cb_embeddding(None, None, None, None,  using_upload_file = True, process_cb_idx=[0])
            self._save_checkpoint(epoch=0, ckpt_file="init_state_tree")
        elif self.args.init_method in ["kmeans_init"]:
            self.model.eval()
            if self.args.target_codebook_generate_method == "system":
                self.model.init_target_cb_embeddding(data, 
                                                     labels_mapping, 
                                                     self.args.column_name, 
                                                     self.ckpt_dir, 
                                                     using_upload_file = False, 
                                                     process_cb_idx=self.args.init_class_embedding_codebook_level, 
                                                     target_datasets_path=self.args.target_codebook_system_datasets,
                                                     remove_first_label=((self.args.dataset== "wos" and self.args.filter_dataset_first_id != -1)))
            elif self.args.target_codebook_generate_method == "user":
                self.model.init_target_cb_embeddding(None, None, None, None,  using_upload_file = True, process_cb_idx=self.args.init_class_embedding_codebook_level)
            # else:
            #     raise Exception(f"Wrong target_codebook_generate_method {self.args.target_codebook_generate_method}")
            self.model.vq_initialization(
                data.dataset['labels'], 
                data.dataset[column_name].to(self.device), 
                first_cb_is_initted = False,
                using_kmeans = True)
            self._save_checkpoint(epoch=0, ckpt_file="init_state_tree")
        else:
            raise Exception("**************** You donot init codebook first!!! ****************")


        #第二步：开始训练
        cur_eval_step = 0
        old_save = None
        for epoch_idx in range(self.epochs):
            if epoch_idx == 0:
                res_dict, num_sample = self._valid_epoch(data, epoch_idx, column_name, labels_mapping)

            # train
            training_start_time = time()
            train_loss, train_recon_loss, train_quant_loss, train_custom_loss = self._train_epoch(data, epoch_idx, column_name)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss, train_quant_loss,
                train_custom_loss
            )
            self.logger.info(train_loss_output)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                res_dict, num_sample = self._valid_epoch(test_data, epoch_idx, column_name, labels_mapping)
                # res_dict, num_sample = self._valid_epoch(data, epoch_idx, column_name, labels_mapping)
                cb_precision_1st, cb_precision_2nd, cb_precision_3rd = [num / num_sample for num in res_dict['correct_sample_num']]
                distincted_class_num_1st, distincted_class_num_2nd, distincted_class_num_3rd = res_dict['distinct_class_num']
                codebook_order_1st, codebook_order_2nd, codebook_order_3rd = res_dict['codebook_order']
        
                # if train_loss < self.best_loss:
                #     self.best_loss = train_loss
                    # ckpt_path = self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)
                ckpt_path = ''
                
                if cb_precision_1st > self.best_precision or cb_precision_2nd > self.best_secondlevel_precision or cb_precision_3rd > self.best_thirdlevel_precision:
                    # if old_save is not None:
                    #     delete_file(old_save)
                    self.best_precision = max(self.best_precision, cb_precision_1st)
                    self.best_secondlevel_precision = max(self.best_secondlevel_precision, cb_precision_2nd)
                    self.best_thirdlevel_precision = max(self.best_thirdlevel_precision, cb_precision_3rd)
                    cur_eval_step = 0
                    # ckpt_path = self._save_checkpoint(epoch_idx, precision=precision,
                    #                       ckpt_file=self.best_precision_ckpt)
                    ckpt_path = self._save_checkpoint(epoch_idx,
                                                      precision=cb_precision_1st,
                                                      secondlevel_precision=cb_precision_2nd,
                                                      thirdlevel_precision=cb_precision_3rd)
                    old_save=ckpt_path
                else:
                    cur_eval_step += 1


                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("cb_precision_1st/2nd", "blue")
                    + ": %f / %f / %f,  "
                    + set_color("distincted_class_num_1st/2nd", "blue")
                    + ":%d / %d / %d,"
                    + set_color("cborder_1st", "blue")
                    + ":%s,"
                    + set_color("cborder_2nd", "blue")
                    + ":%s,"
                    + set_color("cborder_3rd", "blue")
                    + ":%s]"
                ) % (epoch_idx, cb_precision_1st, cb_precision_2nd, cb_precision_3rd, \
                    distincted_class_num_1st, distincted_class_num_2nd, distincted_class_num_3rd, \
                    codebook_order_1st, codebook_order_2nd, codebook_order_3rd)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx,
                                                precision=cb_precision_1st,
                                                secondlevel_precision=cb_precision_2nd,
                                                thirdlevel_precision=cb_precision_3rd)
                now_save = (cb_precision_1st, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if cb_precision_1st > -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])



        return self.best_loss, self.best_precision, self.best_secondlevel_precision, self.best_thirdlevel_precision
