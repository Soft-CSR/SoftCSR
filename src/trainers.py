# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, RandomSampler
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, NTXent,GRU4per
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        #self.model_per = GRU4per
        self.online_similarity_model = args.online_similarity_model

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        #projection head for contrastive learn task
        self.projection = nn.Sequential(nn.Linear(self.args.max_seq_length*self.args.hidden_size, \
                                        512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), 
                                        nn.Linear(512, self.args.hidden_size, bias=True))
        if self.cuda_condition:
            self.model.cuda()
            #self.model_per.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)

        # max_optimizer
        for i in self.model.parameters():
            i.requires_grad = False
        for i in self.model.GRU4per.parameters():
            i.requires_grad = True
        self.max_optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, betas=betas,
                          weight_decay=self.args.weight_decay)

        # min_optimizer
        for i in model.parameters():
            i.requires_grad = True
        for i in model.GRU4per.parameters():
            i.requires_grad = False
        self.min_optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, betas=betas,
                          weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        # self.cf_criterion = NTXent()
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

        
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score1(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "HIT@5": eval('{:.4f}'.format(recall[0])), "HIT@10": eval('{:.4f}'.format(recall[1])),"HIT@20": eval('{:.4f}'.format(recall[2])),
            "NDCG@5": eval('{:.4f}'.format(ndcg[0])), "NDCG@10": eval('{:.4f}'.format(ndcg[1])), "NDCG@20": eval('{:.4f}'.format(ndcg[2]))
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')

        return [recall[0], recall[1], recall[2],ndcg[0],ndcg[1], ndcg[2]], str(post_fix)
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
    def item_seq_len(self,input_ids):
        a = []
        for i in input_ids:
            a.append(len(torch.nonzero(i)))
        a = torch.tensor(a)
        return a
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)

        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.max_seq_length).float()

        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class SoftCSRTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):
        super(SoftCSRTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, 
            args)

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model.forward(cl_batch,[0])
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0], 
                                cl_output_slice[1])

        return cl_loss

    def adv_project(self,grad, norm_type='inf', eps=1e-6):
        """
        L0,L1,L2
        """
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction
    def _one_pair_origin_augmentation_sequence_min(self, sequence_output_origin,inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        origin_output_slice = sequence_output_origin.view(sequence_output_origin.shape[0], -1)
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)

        cl_sequence_output = self.model.forward(cl_batch, [0])

        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0] // 2  # 256
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
 
        cl_loss_1 = self.cf_criterion(origin_output_slice,cl_output_slice[0])

        cl_loss_2 = self.cf_criterion(origin_output_slice,cl_output_slice[1])

        cl_OA_loss = self.args.alpha_1 * cl_loss_1 + self.args.alpha_2 * cl_loss_2

        return cl_OA_loss

    def iteration(self, epoch, dataloader, full_sort=True, train=True):
        str_code = "train" if train else "test"
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")  # 140
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                # Outer minimization:
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ----------- recommendation task ----------#
                sequence_output = self.model.forward(input_ids, [0])
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                # ---------- contrastive learning task with augmentation sequence -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    cl_losses.append(cl_loss)
                joint_loss_cl_oa = 0
                for cl_loss in cl_losses:
                    joint_loss_cl_oa += self.args.alpha * cl_loss

                # ---------- contrastive learning task origin sequence and augmentation sequence-------------#
                sequence_output_origin = sequence_output
                cl_OA_loss = []
                for cl_batch in cl_batches:
                    cl_OA = self._one_pair_origin_augmentation_sequence_min(sequence_output_origin,cl_batch)
                    cl_OA_loss.append(cl_OA)
                for cl_OA in cl_OA_loss:
                    joint_loss_cl_oa += cl_OA
                joint_loss_cl_oa += rec_loss
                joint_loss_cl_oa.backward(retain_graph=True)

                # inner maximize
                for i in range(self.args.adv_step):
                    for cl_batch in cl_batches:
                        joint_loss_at = 0
                        cl_batch = torch.cat(cl_batch, dim=0)
                        cl_batch = cl_batch.to(self.device)

                        # method 1: add perturbation in sequence after encode
                        if self.args.method_sequence == "Yes":
                            cl_sequence_output = self.model.forward(cl_batch,[0])
                            cl_sequence_output = torch.sum(cl_sequence_output.view(cl_sequence_output.shape[0], -1), dim=-1)

                            per_0 = cl_sequence_output.new(cl_sequence_output.size()).normal_(0, 1) * 1e-5
                            per_0.requires_grad_()
                            perturbed_sequence = cl_sequence_output.data + per_0

                            cl_sequence_flatten = perturbed_sequence.view(cl_batch.shape[0], -1)

                            batch_size = cl_batch.shape[0] // 2
                            cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
                            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1])

                            origin_output_slice = torch.sum(sequence_output_origin.view(sequence_output_origin.shape[0], -1),dim=-1).unsqueeze(1)
                            cl_loss_1 = self.cf_criterion(cl_output_slice[0],origin_output_slice)
                            cl_loss_2 = self.cf_criterion(cl_output_slice[1],origin_output_slice)
                            cl_OA_loss = self.args.beta_1 * cl_loss_1 + self.args.beta_2 * cl_loss_2
                            joint_loss_at = self.args.beta_0 * cl_loss + cl_OA_loss

                            delta_grad_0, = torch.autograd.grad(joint_loss_at, per_0, only_inputs=True)
                            norm_0 = delta_grad_0.norm()
                            if torch.isnan(norm_0) or torch.isinf(norm_0):
                                return None
                            # inner sum
                            per_0 = per_0 + delta_grad_0 * 1e-3
                            item_per = per_0
                            item_per = self.adv_project(item_per, norm_type=self.args.norm_type, eps=self.args.epsilon_sequence)

                            perturbed_sequence = cl_sequence_output + self.args.eta * item_per.detach()

                            # train Again
                            cl_sequence_flatten = perturbed_sequence.view(cl_batch.shape[0], -1)
                            batch_size = cl_batch.shape[0] // 2
                            cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
                            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1])
                            cl_loss_1 = self.cf_criterion(cl_output_slice[0], origin_output_slice)
                            cl_loss_2 = self.cf_criterion(cl_output_slice[1], origin_output_slice)
                            cl_OA_loss = self.args.beta_1 * cl_loss_1 + self.args.beta_2 * cl_loss_2
                            joint_loss_at = self.args.beta_0 * cl_loss + cl_OA_loss
                            joint_loss_at.backward(retain_graph=True)

                        # method 2: add perturbation in item of sequence
                        if self.args.method_item == "Yes":
                            cl_item_output = self.model.item_embeddings(cl_batch)
                            per_0 = cl_item_output.new(cl_item_output.size()).normal_(0, 1) * 1e-5

                            per_0.requires_grad_()
                            new_cl_sequence_output = cl_item_output.data + per_0
                            perturbed_sequence = self.model.forward(cl_batch, new_cl_sequence_output)
                            cl_sequence_flatten = perturbed_sequence.view(cl_batch.shape[0], -1)
                            batch_size = cl_batch.shape[0] // 2
                            cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
                            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1])
                            origin_output_slice = sequence_output_origin.view(sequence_output_origin.shape[0], -1)
                            cl_loss_1 = self.cf_criterion(cl_output_slice[0], origin_output_slice)
                            cl_loss_2 = self.cf_criterion(cl_output_slice[1], origin_output_slice)
                            cl_OA_loss = self.args.beta_1 * cl_loss_1 + self.args.beta_2 * cl_loss_2
                            joint_loss_at = self.args.beta_0 * cl_loss + cl_OA_loss
                            delta_grad_0, = torch.autograd.grad(joint_loss_at, per_0, only_inputs=True)
                            norm_0 = delta_grad_0.norm()

                            if torch.isnan(norm_0) or torch.isinf(norm_0):
                                return None

                            per_0 = per_0 + delta_grad_0 * self.args.eta
                            item_per = per_0
                            item_per = self.adv_project(item_per, norm_type=self.args.norm_type, eps=self.args.epsilon_item)

                            perturbed_item_gru = cl_item_output + self.args.eta * item_per.detach()

                            # train Again
                            perturbed_sequence = self.model.forward(cl_batch, perturbed_item_gru)
                            cl_sequence_flatten = perturbed_sequence.view(cl_batch.shape[0], -1)
                            batch_size = cl_batch.shape[0] // 2
                            cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
                            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1])
                            origin_output_slice = sequence_output_origin.view(sequence_output_origin.shape[0], -1)
                            cl_loss_1 = self.cf_criterion(cl_output_slice[0], origin_output_slice)
                            cl_loss_2 = self.cf_criterion(cl_output_slice[1], origin_output_slice)
                            cl_OA_loss = self.args.beta_1 * cl_loss_1 + self.args.beta_2 * cl_loss_2
                            joint_loss_at = self.args.beta_0 * cl_loss + cl_OA_loss

                            joint_loss_at.backward(retain_graph=True)

                        # method 3: add perturbation in item by model gru
                        if self.args.method_gru == "Yes":
                            cl_item_output = self.model.item_embeddings(cl_batch)
                            item_per = self.model.GRU4per(cl_item_output) * 1e-5
                            item_per = item_per / (torch.norm(item_per, dim=-1, keepdim=True) + self.args.sigma_gru)
                            perturbed_item_gru = cl_item_output + self.args.gamma_gru * item_per.detach()
                            # train Again
                            perturbed_sequence = self.model.forward(cl_batch, perturbed_item_gru)
                            cl_sequence_flatten = perturbed_sequence.view(cl_batch.shape[0], -1)
                            batch_size = cl_batch.shape[0] // 2
                            cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
                            cl_loss = self.cf_criterion(cl_output_slice[0],cl_output_slice[1])
                            origin_output_slice = sequence_output_origin.view(sequence_output_origin.shape[0], -1)
                            cl_loss_1 = self.cf_criterion(cl_output_slice[0],
                                                          origin_output_slice)
                            cl_loss_2 = self.cf_criterion(cl_output_slice[1],
                                                          origin_output_slice)
                            cl_OA_loss = self.args.beta_1 * cl_loss_1 + self.args.beta_2 * cl_loss_2
                            joint_loss_at = self.args.beta_0 * cl_loss + cl_OA_loss
                            joint_loss_at.backward(retain_graph=True)
                        # ——————————————
                        self.max_optimizer.step()
                        self.max_optimizer.zero_grad()
                        # ——————————————
                self.min_optimizer.step()
                self.min_optimizer.zero_grad()

                rec_avg_loss += rec_loss.item()
                joint_avg_loss += joint_loss_at.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            # for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
            #     post_fix['cl_pair_'+str(i)+'_loss'] = '{:.4f}'.format(cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0: # args.log_freq=1
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch), # Recommendation EP_test:0
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.forward(input_ids, [0])  # [0,0]
                    recommend_output = recommend_output[:, -1, :]
                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)
