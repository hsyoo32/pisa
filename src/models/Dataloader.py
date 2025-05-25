import torch
import logging
import os
import numpy as np
import copy
from random import randint
from torch.utils.data import Dataset as BaseDataset
from utils import utils
from scipy.sparse import csr_matrix
from time import time
import scipy.sparse as sp


class Dataset(BaseDataset):
    def __init__(self, args, corpus, data_type, data_idx):
        # self.model = model  # model object reference
        self.corpus = corpus  # reader object reference
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.time_idx = data_idx
        self.data_type = data_type

        self.train_file = os.path.join(corpus.snapshots_path, data_type+'_block'+str(data_idx))
        
        self.train_data = utils.read_data_from_file_int(self.train_file)
        self.train_data = np.array(self.train_data)

        print('train data shape',self.train_data.shape)


        self.trainUser = self.train_data[:, 0]
        self.trainItem = self.train_data[:, 1]
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
									shape=(corpus.n_users, corpus.n_items))    
        self.Graph = None

        self.user_set = set(self.trainUser)
        self.item_set = set(self.trainItem)
        # self.num_users = len(self.user_set)
        # self.num_items = len(self.item_set)
        self.num_users = self.trainUser.max() + 1
        self.num_items = self.trainItem.max() + 1
        # get neighbor set for each user
        self.user_neigh = {}
        for user in self.user_set:
            self.user_neigh[user] = list(self.UserItemNet[user].indices)

        
        # get neighbor set for each item
        # self.item_neigh = {}
        # for item in self.item_set:
        #     self.item_neigh[item] = list(self.UserItemNet[:, item].nonzero()[0])
        


    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, index: int) -> dict:
        current = self._get_feed_dict(index)
        #print('index: {}'.format(index))
        return current

    def _get_feed_dict(self, index: int) -> dict:

        user_id, item_id = self.train_data[index]
        neg_items = self._sample_neg_items(user_id).squeeze()
        user_id, item_id = torch.tensor([user_id]), torch.tensor([item_id])
        item_id_ = torch.cat((item_id, neg_items), axis=-1)
        
        feed_dict = {'user_id': user_id, #(batch_size, )
                        'item_id': item_id_} #(batch_size, 1+neg_items)

        return feed_dict
    

    def _sample_neg_items(self, user_id):
        #num_neg = self.model.num_neg
        num_neg = self.args.num_neg


        neg_items = torch.zeros(size=(1, num_neg), dtype=torch.int64)
        #neg_items = torch.zeros(size=(num_neg), dtype=torch.int64)

        #for idx, user in enumerate(self.corpus.user_list[index:index_end]): # Automatic coverage?
        #for idx, user in enumerate(user_id): # Automatic coverage?

        user_clicked_set = copy.deepcopy(self.corpus.user_clicked_set[user_id])
        # By copying, it may not collide with other process with same user index
        for neg in range(num_neg):
            neg_item = self._randint_w_exclude(user_clicked_set)
            neg_items[0][neg] = neg_item
            # Skip below: one neg for train
            user_clicked_set = np.append(user_clicked_set, neg_item)

        return neg_items

    def _randint_w_exclude(self, clicked_set):
        randItem = randint(1, self.corpus.n_items-1)
        return self._randint_w_exclude(clicked_set) if randItem in clicked_set else randItem
    

    # for GCN
    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.n_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self._device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            adj_mat_path = os.path.join(self.corpus.snapshots_path, self.data_type+'_adj_mat_t{}.npz'.format(self.time_idx))
            try:
                pre_adj_mat = sp.load_npz(adj_mat_path)
                print("successfully loaded adjacency matrix...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.corpus.n_users + self.corpus.n_items, self.corpus.n_users + self.corpus.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.corpus.n_users, self.corpus.n_users:] = R
                adj_mat[self.corpus.n_users:, :self.corpus.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"computing time {end-s}s, save norm_mat..")
                sp.save_npz(adj_mat_path, norm_adj)

            # if self.split == True:
            #     self.Graph = self._split_A_hat(norm_adj)
            #     print("done split matrix")
            # else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self._device)
                #print("don't split the matrix")
        return self.Graph