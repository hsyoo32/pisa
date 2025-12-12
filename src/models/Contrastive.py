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
    runner = 'Runner_cons'
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
        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            #torch.nn.init.normal_(m.weight, mean=0.0, std=1)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
                #torch.nn.init.normal_(m.bias, mean=0.0, std=1)
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
        self.weights_1 = None
        self.weights_2 = None
        self.centr = None
        self.centr_prev = None
        self.Gu = None
        self.Gu_prev = None
        self.test_result_file = args.test_result_file

        self._define_params()

        if self.time_idx > 0:
            self._define_params_piw()



        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)


    def _define_params_piw(self):
        self.W = nn.Parameter(torch.empty(self.cluster_num, self.emb_size, self.emb_size))
        self.W_prev = nn.Parameter(torch.empty(self.cluster_num, self.emb_size, self.emb_size))
        #self.W_hist = nn.Parameter(torch.empty(self.cluster_num, self.emb_size, self.emb_size))

        #if 'stdev' in self.dyn_method:
        nn.init.normal_(self.W, mean=0.0, std=1)
        nn.init.normal_(self.W_prev, mean=0.0, std=1)

        self.add_drop = nn.Sequential(
            nn.Linear(self.cluster_num, self.hidden_layer),  # M -> L
            nn.ReLU(),  # ReLU activation function
            nn.Linear(self.hidden_layer, 1)  # L -> 1
            )


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

        self.centr = centr
        self.centr_prev = centr_prev  

    def generate_weight(self, u_vectors, u_vectors_prev, cent, cent_prev, option='softplus'):
        h = u_vectors.t()
        h_prev = u_vectors_prev.t()

        Gu = torch.empty((self.cluster_num, h.shape[1]), device=self._device)
        Gu_prev = torch.empty((self.cluster_num, h_prev.shape[1]), device=self._device)

        for i in range(self.cluster_num):
            uWh = torch.matmul(cent[i], self.W[i]) @ h
            Gu[i] = uWh
            #Gu[i] = torch.exp(uWh) / torch.exp(uWh).sum(dim=0)
            uWh_prev = torch.matmul(cent[i], self.W[i]) @ h_prev
            Gu_prev[i] = uWh_prev
            #Gu_prev[i] = torch.exp(uWh_prev) / torch.exp(uWh_prev).sum(dim=0)

        Gu = F.softmax(Gu, dim=0)
        Gu_prev = F.softmax(Gu_prev, dim=0)

        #Su = torch.transpose((Gu_prev - Gu).pow(2), 0, 1).to(self._device)
        #Su = torch.transpose(torch.abs(Gu_prev - Gu), 0, 1).to(self._device)
        scaling_factor = 10**0
        gamma = 5
        
        #Su = torch.transpose(((Gu_prev - Gu).pow(2)+1.5)**gamma, 0, 1).to(self._device)
        Su = torch.transpose(((Gu_prev - Gu).pow(2)), 0, 1).to(self._device)

            
            

        if 'softplus' in option:
            x = self.add_drop(Su)
            x1 = nn.Softplus()(x) # (B' X 1)
            x2 = nn.Softplus()(-x)

            # if self.logging_flag == 0:
            #     for i in range(5):
            #         logging.info('x[{}]: {}, x1[{}]: {}, x2[{}]: {}'.format(i, x[i].item(), i, x1[i].item(), i, x2[i].item()))
            #         #logging.info('x1[{}]: {}, x2[{}]: {}'.format(i, x1[i].item(), i, x2[i].item()))

            return x1, x2
 

    def return_zero_losses(self, bpr_loss):
        zero_tensor = torch.tensor([0.0], dtype=torch.float32).to(self._device)
        return bpr_loss, bpr_loss, zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor, None

    def loss(self, data, batch_data, prev_data, time_idx, prev_model, reduction):
        all_users, all_items = self.computer()
        u_ids = batch_data['user_id'].repeat((1, batch_data['item_id'].shape[1])).to(torch.long)
        i_ids = batch_data['item_id'].to(torch.long)
        
        u_vectors, i_vectors = all_users[u_ids], all_items[i_ids]
        predictions = (u_vectors * i_vectors).sum(dim=-1)

        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:1 + self.num_neg]
        bpr_loss = -(pos_pred[:, None] - neg_pred).sigmoid().log().mean(dim=1)
        bpr_loss = bpr_loss.mean() if reduction == 'mean' else bpr_loss

        if time_idx == 0:
            return self.return_zero_losses(bpr_loss)

        
        # unique users and items that are both in the current and previous data

        pos_i_ids = i_ids[:, 0]
        users, items = u_ids.unique(), pos_i_ids.unique()
        users = users[torch.isin(users, torch.tensor(list(prev_data.user_set)).to(self._device))]
        items = items[torch.isin(items, torch.tensor(list(prev_data.item_set)).to(self._device))]
        if len(users) == 0 or len(items) == 0:
            return self.return_zero_losses(bpr_loss)
        
        # for user in users:
        #     neighbor_items = prev_data.user_neigh[user]

        
        # distillation on each layer (embedding) of the lightgcn model 
        if 'layerwise' in self.dyn_method:
            _, _, all_users_layer, all_items_layer = self.computer_layer()
            _, _, all_users_layer_prev, all_items_layer_prev = prev_model.computer_layer()

        else:
            all_users_prev, all_items_prev = prev_model.computer()
            u_vectors_prev, i_vectors_prev = all_users_prev[users], all_items_prev[items]
            u_vectors, i_vectors = all_users[users], all_items[items]

        # distillation between user in the previous time & user in the current time
        # distillation between user's neighbors in the previous time & user in the current time 
        neighbor_items_total = torch.tensor([]).to(torch.long).to(self._device)
        users_total = torch.tensor([]).to(torch.long).to(self._device)
        for user in users:
            neighbor_items = prev_data.user_neigh[user.item()]
            if len(neighbor_items) == 0:
                continue
            neighbor_items = torch.tensor(neighbor_items).to(self._device)
            neighbor_items_total = torch.cat((neighbor_items_total, neighbor_items), axis=0)
            # duplicate user as same size as the number of neighbor items
            users_total = torch.cat((users_total, user.repeat((len(neighbor_items)))), axis=0)


        neighbor_users_total = torch.tensor([]).to(torch.long).to(self._device)
        items_total = torch.tensor([]).to(torch.long).to(self._device)
        for item in items:
            neighbor_users = prev_data.item_neigh[item.item()]
            if len(neighbor_users) == 0:
                continue
            neighbor_users = torch.tensor(neighbor_users).to(self._device)
            neighbor_users_total = torch.cat((neighbor_users_total, neighbor_users), axis=0)
            items_total = torch.cat((items_total, item.repeat((len(neighbor_users)))), axis=0)

        if 'layerwise' in self.dyn_method:
                  
            kd_loss_user_list, kd_loss_item_list, kd_loss_user_neighbor_list, kd_loss_item_neighbor_list = [], [], [], []
            for layer in range(all_users_layer.shape[1]):
                all_users, all_items = all_users_layer[:, layer], all_items_layer[:, layer]
                all_users_prev, all_items_prev = all_users_layer_prev[:, layer], all_items_layer_prev[:, layer]
                u_vectors, i_vectors = all_users[users], all_items[items]
                u_vectors_prev, i_vectors_prev = all_users_prev[users], all_items_prev[items]

                kd_loss_user, kd_loss_item, kd_loss_user_neighbor, kd_loss_item_neighbor = self.get_kd_losses(
                    users, u_vectors, u_vectors_prev, i_vectors, i_vectors_prev, all_users, all_items, 
                    all_users_prev, all_items_prev, users_total, items_total, neighbor_users_total, neighbor_items_total)
                
                kd_loss_user_list.append(kd_loss_user)
                kd_loss_item_list.append(kd_loss_item)
                kd_loss_user_neighbor_list.append(kd_loss_user_neighbor)
                kd_loss_item_neighbor_list.append(kd_loss_item_neighbor)

            kd_loss_user = sum(kd_loss_user_list) / len(kd_loss_user_list)
            kd_loss_item = sum(kd_loss_item_list) / len(kd_loss_item_list)
            kd_loss_user_neighbor = sum(kd_loss_user_neighbor_list) / len(kd_loss_user_neighbor_list)
            kd_loss_item_neighbor = sum(kd_loss_item_neighbor_list) / len(kd_loss_item_neighbor_list)

                

        else:
            # user-side
            if 'piw' in self.dyn_method:

                weights_1, _ = self.generate_weight(u_vectors, u_vectors_prev, self.centr, self.centr_prev, 'softplus')


                # if self.logging_flag == 0:
                #     for i in range(5):
                #         logging.info('weights_1[{}]: {:.5f}'.format(i, weights_1[i].item()))
                #     self.logging_flag = 1   


                kd_loss_user = self.condition_info_nce_for_embeddings(u_vectors_prev, u_vectors, weights=weights_1, tau=0.5)
                user_to_weight = {users[i].item(): weights_1[i].item() for i in range(len(users))}
                #print('user_to_weight: {}'.format(user_to_weight))
                wights_1_neigh = torch.tensor([user_to_weight[user.item()] for user in users_total]).to(self._device)


                kd_loss_user_neighbor = self.condition_info_nce_for_embeddings(all_items_prev[neighbor_items_total], all_users[users_total], weights=wights_1_neigh, tau=0.5)

            else:
                kd_loss_user = self.condition_info_nce_for_embeddings(u_vectors_prev, u_vectors, weights=None, tau=0.5)
                kd_loss_user_neighbor = self.condition_info_nce_for_embeddings(all_items_prev[neighbor_items_total], all_users[users_total], weights=None, tau=0.5)
                
            # item-side
            kd_loss_item = self.condition_info_nce_for_embeddings(i_vectors_prev, i_vectors, weights=None, tau=0.5)
            kd_loss_item_neighbor = self.condition_info_nce_for_embeddings(all_users_prev[neighbor_users_total], all_items[items_total], weights=None, tau=0.5)



        if 'onlyuser' in self.dyn_method:
            kd_loss = kd_loss_user + kd_loss_user_neighbor
        else:
            kd_loss = kd_loss_user + kd_loss_item + kd_loss_user_neighbor + kd_loss_item_neighbor
        
        kd_loss = kd_loss * self.bound_weight   
        loss = bpr_loss + kd_loss


        
        return loss, bpr_loss, kd_loss, kd_loss_user, kd_loss_item, kd_loss_user_neighbor, kd_loss_item_neighbor


    def get_kd_losses(self, users, u_vectors, u_vectors_prev, i_vectors, i_vectors_prev, all_users, all_items, all_users_prev, all_items_prev, 
                      users_total, items_total, neighbor_users_total, neighbor_items_total):

        if 'piw' in self.dyn_method:

            weights_1, _ = self.generate_weight(u_vectors, u_vectors_prev, self.centr, self.centr_prev, 'softplus')


            kd_loss_user = self.condition_info_nce_for_embeddings(u_vectors_prev, u_vectors, weights=weights_1, tau=0.5)
            user_to_weight = {users[i].item(): weights_1[i].item() for i in range(len(users))}
            #print('user_to_weight: {}'.format(user_to_weight))
            wights_1_neigh = torch.tensor([user_to_weight[user.item()] for user in users_total]).to(self._device)


            kd_loss_user_neighbor = self.condition_info_nce_for_embeddings(all_items_prev[neighbor_items_total], all_users[users_total], weights=wights_1_neigh, tau=0.5)

        else:
            kd_loss_user = self.condition_info_nce_for_embeddings(u_vectors_prev, u_vectors, weights=None, tau=0.5)
            kd_loss_user_neighbor = self.condition_info_nce_for_embeddings(all_items_prev[neighbor_items_total], all_users[users_total], weights=None, tau=0.5)
            
        # item-side
        kd_loss_item = self.condition_info_nce_for_embeddings(i_vectors_prev, i_vectors, weights=None, tau=0.5)
        kd_loss_item_neighbor = self.condition_info_nce_for_embeddings(all_users_prev[neighbor_users_total], all_items[items_total], weights=None, tau=0.5)

        return kd_loss_user, kd_loss_item, kd_loss_user_neighbor, kd_loss_item_neighbor


    def condition_info_nce_for_embeddings(self, x, z, weights=None, tau=0.5):
        # lower bound of the mutual information
        # x: previous embeddings (anchor) (B' X D)
        # z: current embeddings (B' X D)
        # tau: temperature
        # use batch current embeddings as negative samples
        # weights: personalized weights for each user (B' X 1)
        x_norm = F.normalize(x)
        y_norm = F.normalize(z)

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

