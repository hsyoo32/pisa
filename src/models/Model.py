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
from torch.nn import functional as F
from torch import nn

DEFAULT_EPS = 1e-10

class Model(torch.nn.Module):
    reader = 'Reader'
    runner = 'Runner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--num_neg', type=int, default=4,
                            help='The number of negative items for training.')
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
        self.item_num = corpus.n_items
        self.user_num = corpus.n_users
        self.optimizer = None
        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info('#params: %d' % self.total_parameters)
        self.dyn_method = args.dyn_method
        self.test_result_file = args.test_result_file

    def get_relevances(self, model, user, items):
        pred_eval = model.model_(user, items, self.DRM)

        return pred_eval.cpu().data.numpy()

    def loss(self, current_data, reduction):
        all_users, all_items = self.computer()
        u_ids = current_data['user_id'].repeat((1, current_data['item_id'].shape[1])).to(torch.long)
        i_ids = current_data['item_id'].to(torch.long)
        u_vectors = all_users[u_ids]
        i_vectors = all_items[i_ids]
        predictions = (u_vectors * i_vectors).sum(dim=-1)

        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:1 + self.num_neg]  # 1 pos : self.num_neg neg
        # BPR loss
        bpr_loss = -(pos_pred[:, None] - neg_pred).sigmoid().log().mean(dim=1)
        if reduction == 'mean':
            bpr_loss = bpr_loss.mean()

        return bpr_loss

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
            
        self.load_state_dict(check_point['model_state_dict'])

        return check_point

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

