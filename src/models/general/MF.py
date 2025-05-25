# -*- coding: UTF-8 -*-
import torch.nn as nn
from models.Model import Model
import torch

class MF(Model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return Model.parse_model_args(parser)

    def __init__(self, args, corpus, data):
        self.emb_size = args.emb_size
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def forward(self, u_ids, i_ids, flag):
        u_ids = u_ids.repeat((1, i_ids.shape[1]))  # [batch_size, -1]
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
            
        #return prediction.view(len(u_ids), -1)
        return prediction
    
    # inference
    def infer_user_scores(self, u_ids, i_ids):
        #u_ids = u_ids.repeat((1, i_ids.shape[1]))
        # print('u_ids shape:', u_ids.shape)
        # print('i_ids shape:', i_ids.shape)
        
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        scores = torch.matmul(cf_u_vectors, cf_i_vectors.t())

        return scores
    






