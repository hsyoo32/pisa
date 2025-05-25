# -*- coding: UTF-8 -*-

import torch
import logging
import os
import numpy as np
import copy
from random import randint
from torch.utils.data import Dataset as BaseDataset
from typing import NoReturn, List

from utils import utils
from helpers.Reader import Reader
from collections import defaultdict
import torch.nn as nn
# from models.Club import CLUBSample
import torch.nn.functional as F

DEFAULT_EPS = 1e-10

class Model(torch.nn.Module):
    reader = 'Reader'
    runner = 'Runner_PISA'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--num_neg', type=int, default=4,
                            help='The number of negative items for training.')
        parser.add_argument('--num_neg_fair', type=int, default=4,
                            help='The number of negative items for the fairness loss')
        parser.add_argument('--DRM', type=str, default='',
                            help='Use DRM regularization or not.')
        parser.add_argument('--DRM_weight', type=float, default=1,
                            help='DRM term weight.')
        parser.add_argument('--tau', type=float, default=3.0,
                            help='DRM hyperparameter tau.')
        parser.add_argument('--kd', type=int, default=1,
                            help='enable knowledge distillation')
        parser.add_argument('--cluster_num', type=int, default=20,
                            help='')
        parser.add_argument('--hidden_layer', type=int, default=50,
                            help='')
        parser.add_argument('--bound_weight', type=float, default=0.1,
                            help='')
        parser.add_argument('--ratio', type=float, default=0.5,
                            help='')
        parser.add_argument('--temp', type=float, default=1.0,
                            help='')
        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)


    def __init__(self, args, corpus: Reader):
        super(Model, self).__init__()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = args.model_path
        self.num_neg = args.num_neg
        self.num_neg_fair = args.num_neg_fair
        self.item_num = corpus.n_items
        self.user_num = corpus.n_users
        self.optimizer = None
        self.dyn_method = args.dyn_method
        self.cluster_num = args.cluster_num
        self.hidden_layer = args.hidden_layer
        self.bound_weight = args.bound_weight
        self.ratio = args.ratio
        self.logging_flag = 0
        self.epoch = 0
        self.freeze_flag = 0
        self.temp = args.temp
        self.test_result_file = args.test_result_file
        self.forward_flag = 0

        self._define_params()

        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)


    def kmeans(self, X, K=20, max_iters=100, tol=1e-4, reinit_empty_clusters=True):
        m, n = X.shape
        centroids = X[torch.randperm(m)[:K]]

        for _ in range(max_iters):
            distances = torch.cdist(X, centroids)
            clusters = torch.argmin(distances, dim=1)

            new_centroids = [X[clusters == k].mean(dim=0) if (clusters == k).shape[0] > 0 else centroids[k] for k in range(K)]

            if reinit_empty_clusters:
                for k in range(K):
                    if (clusters == k).sum() == 0:  # Reinitialize empty cluster
                        print('Reinitialize empty cluster')
                        new_centroids[k] = X[torch.randint(0, m, (1,))].squeeze()

            new_centroids = torch.stack(new_centroids)

            if torch.all(torch.norm(new_centroids - centroids, dim=1) < tol):
                break

            centroids = new_centroids

        clusters_points = [torch.nonzero(clusters == k).squeeze() for k in range(K)]
        return clusters, centroids, clusters_points
    

    def update_kmeans(self, prev_model):
        k = self.cluster_num

        all_users, all_items = self.computer()
        all_users_prev, all_items_prev = prev_model.computer()
        points_item = []
        points_item_prev = []
        i = 0
        while len(points_item) != len(points_item_prev) or i == 0:
            cluster_item, centr, points_item = self.kmeans(all_items, K=k-i*2, max_iters=100)
            cluster_item_prev, centr_prev, points_item_prev = self.kmeans(all_items_prev, K=k-i*2, max_iters=100)
            i += 1
            # if i > 1:
            #     print('reinitialize ######################## {}'.format(i))

        self.centr = centr
        self.centr_prev = centr_prev  


    def generate_deterministic_weight(self, u_vectors, u_vectors_prev, cent, cent_prev):
        h = u_vectors.t()
        h_prev = u_vectors_prev.t()

        Gu = torch.empty((self.cluster_num, h.shape[1]), device=self._device)
        Gu_prev = torch.empty((self.cluster_num, h_prev.shape[1]), device=self._device)

        for i in range(self.cluster_num):
            Gu[i] = cent[i] @ h
            Gu_prev[i] = cent[i] @ h_prev

        Gu = torch.transpose(F.softmax(Gu, dim=0), 0, 1) # (B' X K)
        Gu_prev = torch.transpose(F.softmax(Gu_prev, dim=0), 0, 1) # (B' X K)

        def jsd(p, q):
            m = 0.5 * (p + q)
            return 0.5 * (torch.sum(p * (torch.log(p + DEFAULT_EPS) - torch.log(m + DEFAULT_EPS)), dim=1) + 
                  torch.sum(q * (torch.log(q + DEFAULT_EPS) - torch.log(m + DEFAULT_EPS)), dim=1))

        weight = jsd(Gu_prev, Gu)
        mean = torch.mean(weight)
        weight = weight - mean
        weight = torch.sigmoid(weight)

        return weight

    def return_zero_losses(self, bpr_loss):
        zero_tensor = torch.tensor([0.0], dtype=torch.float32).to(self._device)
        return bpr_loss, bpr_loss, zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor

    def loss(self, data, current_data, prev_data, time_idx, prev_model, forward_model, reduction):
        all_users, all_items = self.computer()
        u_ids = current_data['user_id'].repeat((1, current_data['item_id'].shape[1])).to(torch.long)
        i_ids = current_data['item_id'].to(torch.long)
        
        u_vectors, i_vectors = all_users[u_ids], all_items[i_ids]
        predictions = (u_vectors * i_vectors).sum(dim=-1)

        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:1 + self.num_neg]
        bpr_loss = -(pos_pred[:, None] - neg_pred).sigmoid().log().mean(dim=1)
        bpr_loss = bpr_loss.mean() if reduction == 'mean' else bpr_loss

        if time_idx == 0 or (self.forward_flag == 0 and 'plasticity' in self.dyn_method):
            return self.return_zero_losses(bpr_loss)

        # unique users and items that are both in the current and previous data
        pos_i_ids = i_ids[:, 0]
        users, items = u_ids.unique(), pos_i_ids.unique()
        users = users[torch.isin(users, torch.tensor(list(prev_data.user_set)).to(self._device))]
        items = items[torch.isin(items, torch.tensor(list(prev_data.item_set)).to(self._device))]
        if len(users) == 0 or len(items) == 0:
            return self.return_zero_losses(bpr_loss)

        # Get embeddings
        all_users_prev, all_items_prev = prev_model.computer()
        u_vectors_prev, i_vectors_prev = all_users_prev[users], all_items_prev[items]
        u_vectors, i_vectors = all_users[users], all_items[items]

        if self.forward_flag > 0:
            all_users_forward, all_items_forward = forward_model.computer()
            u_vectors_forward, i_vectors_forward = all_users_forward[users], all_items_forward[items]

        # Neighbor information
        if 'userneigh' in self.dyn_method:
            if 'stability' in self.dyn_method:
                user_neighbors_total = torch.tensor([]).to(torch.long).to(self._device)
                users_total = torch.tensor([]).to(torch.long).to(self._device)
                for user in users:
                    user_neighbors = prev_data.user_neigh[user.item()]
                    if len(user_neighbors) == 0:
                        continue
                    user_neighbors = torch.tensor(user_neighbors).to(self._device)
                    user_neighbors_total = torch.cat((user_neighbors_total, user_neighbors), axis=0)
                    users_total = torch.cat((users_total, user.repeat(len(user_neighbors))), axis=0)

            if 'plasticity' in self.dyn_method:
                user_neighbors_total_new = torch.tensor([]).to(torch.long).to(self._device) # user neighbors (references)
                users_total_new = torch.tensor([]).to(torch.long).to(self._device) # users
                for user in u_ids.unique():
                    user_neighbors = data.user_neigh[user.item()]
                    if len(user_neighbors) == 0:
                        continue
                    user_neighbors = torch.tensor(user_neighbors).to(self._device)
                    user_neighbors_total_new = torch.cat((user_neighbors_total_new, user_neighbors), axis=0) 
                    users_total_new = torch.cat((users_total_new, user.repeat(len(user_neighbors))), axis=0)

                if 'userneighcur' in self.dyn_method:
                    user_neighbors_total_current = torch.tensor([]).to(torch.long).to(self._device)
                    users_total_current = torch.tensor([]).to(torch.long).to(self._device)
                    for user in users:
                        user_neighbors = data.user_neigh[user.item()]
                        if len(user_neighbors) == 0:
                            continue
                        user_neighbors = torch.tensor(user_neighbors).to(self._device)
                        user_neighbors_total_current = torch.cat((user_neighbors_total_current, user_neighbors), axis=0)
                        users_total_current = torch.cat((users_total_current, user.repeat(len(user_neighbors))), axis=0)

        cl_loss = torch.tensor([0.0], dtype=torch.float32).to(self._device)
        plast_loss = torch.tensor([0.0], dtype=torch.float32).to(self._device)
        stab_loss = torch.tensor([0.0], dtype=torch.float32).to(self._device)
        plast_neigh_loss = torch.tensor([0.0], dtype=torch.float32).to(self._device)
        stab_neigh_loss = torch.tensor([0.0], dtype=torch.float32).to(self._device)


        weights_1 = self.generate_deterministic_weight(u_vectors, u_vectors_prev, self.centr, self.centr_prev)
        weights_2 = 1 - weights_1
        
        L = self.ratio
        # only use weights that are top-L% of the weights for the plasticity_enhancement_loss
        weights_1 = torch.where(weights_1 > torch.quantile(weights_1, 1-L), weights_1, torch.tensor([0.0], dtype=torch.float32).to(self._device))
        # only use weights that are bottom-L% of the weights for the stability_enhancement_loss
        weights_2 = torch.where(weights_2 > torch.quantile(weights_2, 1-L), weights_2, torch.tensor([0.0], dtype=torch.float32).to(self._device))

        # # Print weights
        # if self.logging_flag == 0:
        #     self.logging_flag = 1
        #     for i in range(10):
        #         logging.info('weights_1[{}]: {:.5f}, weights_2[{}]: {:.5f}'.format(i, weights_1[i].item(), i, weights_2[i].item()))

        # Plasticity loss
        if 'plasticity' in self.dyn_method:
            plast_loss = self.condition_info_nce_for_embeddings(u_vectors_forward, u_vectors, weights_1)
            cl_loss += plast_loss * self.bound_weight
            if 'userneigh' in self.dyn_method:
                if 'userneighcur' in self.dyn_method:
                    user_to_weight = {user.item(): weight.item() for user, weight in zip(users, weights_1)}   
                    weights_1_neigh_current = torch.tensor([user_to_weight[user.item()] for user in users_total_current]).to(self._device)
                    plast_neigh_loss = self.condition_info_nce_for_embeddings(all_items_forward[user_neighbors_total_current], all_users[users_total_current], weights_1_neigh_current)
                    cl_loss += plast_neigh_loss * self.bound_weight

                else:
                    user_to_weight = {user.item(): weight.item() for user, weight in zip(users, weights_1)}   
                    for user in users_total_new:
                        if user.item() not in user_to_weight:
                            user_to_weight[user.item()] = 0.5
                    
                    weights_1_neigh_new = torch.tensor([user_to_weight[user.item()] for user in users_total_new]).to(self._device)
                    plast_neigh_loss = self.condition_info_nce_for_embeddings(all_items_forward[user_neighbors_total_new], all_users[users_total_new], weights_1_neigh_new)
                    cl_loss += plast_neigh_loss * self.bound_weight

        # Stability loss
        if 'stability' in self.dyn_method:
            stab_loss = self.condition_info_nce_for_embeddings(u_vectors_prev, u_vectors, weights_2)
            cl_loss += stab_loss * self.bound_weight
            if 'userneigh' in self.dyn_method:
                user_to_weight = {user.item(): weight.item() for user, weight in zip(users, weights_2)}   
                weights_2_neigh = torch.tensor([user_to_weight[user.item()] for user in users_total]).to(self._device)
                stab_neigh_loss = self.condition_info_nce_for_embeddings(all_items_prev[user_neighbors_total], all_users[users_total], weights_2_neigh)
                cl_loss += stab_neigh_loss * self.bound_weight

        loss = bpr_loss + cl_loss

        return loss, bpr_loss, cl_loss, plast_loss, stab_loss, plast_neigh_loss, stab_neigh_loss


    def condition_info_nce_for_embeddings(self, x, z, weights=None, tau=0.5):
        # lower bound of the mutual information
        # x: previous embeddings (anchor) (B' X D)
        # z: current embeddings (B' X D)
        # tau: temperature
        # use batch current embeddings as negative samples
        # weights: personalized weights for each user (B' X 1)
        x_norm = F.normalize(x, dim=-1)
        y_norm = F.normalize(z, dim=-1)

        scores = torch.mm(x_norm, y_norm.t())
        scores_diag = torch.diag(scores)
        numerator = torch.exp(scores_diag / tau)  # positive samples
        denominator = torch.sum(torch.exp(scores / tau), dim=1)
        if weights is not None:
            loss = -torch.log(numerator / denominator) * weights.squeeze()
        else:
            loss = -torch.log(numerator / denominator)
        return loss.mean()

        

    def save_model(self, model_path=None, add_path=None):
        if model_path is None:
            model_path = self.model_path
        if add_path:
            model_path += add_path
        utils.check_dir(model_path)
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    model_path)

    def load_model(self, model_path=None, add_path=None, flag=0):
        if model_path is None:
            model_path = self.model_path
        if add_path:
            model_path += add_path
        
        if torch.cuda.is_available():
            check_point = torch.load(model_path)
        else:
            check_point = torch.load(model_path, map_location=torch.device('cpu'))

        # If there are missing keys, load the model without them (for loading pre-trained models without current model-specific keys)
        model_dict = self.state_dict()
        pretrain_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict}
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)

        return check_point

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

