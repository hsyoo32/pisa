# -*- coding: UTF-8 -*-
import torch.nn as nn
from models.Model import Model
import torch

class LGN(Model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='Number of layers')
        parser.add_argument('--keep_prob', type=float, default=-1)

        return Model.parse_model_args(parser)

    def __init__(self, args, corpus, data, time_idx):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.keep_prob = args.keep_prob
        self.Graph = data.getSparseGraph()
        self.time_idx = time_idx

        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def __dropout(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self):
        # GCN propagation
        users_emb = self.u_embeddings.weight
        items_emb = self.i_embeddings.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.keep_prob > 0:
            g_droped = self.__dropout(self.Graph, self.keep_prob)
        else:
            g_droped = self.Graph
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.user_num, self.item_num])
        return users, items

    def forward(self, u_ids, i_ids):
        all_users, all_items = self.computer()
        u_ids = u_ids.repeat((1, i_ids.shape[1])) 
        u_ids=u_ids.to(torch.long)
        i_ids=i_ids.to(torch.long)
        cf_u_vectors = all_users[u_ids]
        cf_i_vectors = all_items[i_ids]
        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)

        return prediction
    
    # inference
    def infer_user_scores(self, u_ids, i_ids):
        all_users, all_items = self.computer()
        cf_u_vectors = all_users[u_ids]
        cf_i_vectors = all_items[i_ids]
        scores = torch.matmul(cf_u_vectors, cf_i_vectors.t())

        return scores
