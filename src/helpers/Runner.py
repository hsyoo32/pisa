# -*- coding: UTF-8 -*-

import os
import gc
import copy
import torch
import logging
import numpy as np
import random
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Dataloader
from typing import Dict, List, NoReturn

from utils import utils
from models.Model import Model

import matplotlib.pyplot as plt
import Inference
import json


class Runner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--tepoch', type=int, default=10,
                            help='Number of epochs.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=1e-04,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1,
                            help='pin_memory in DataLoader')
        parser.add_argument('--test_result_file', type=str, default='',
                            help='')

        return parser

    def __init__(self, args, corpus):
        self.epoch = args.epoch
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.result_file = args.result_file
        self.dyn_method = args.dyn_method
        self.time = None  # will store [start_time, last_step_time]

        self.snap_boundaries = corpus.snap_boundaries
        self.snapshots_path = corpus.snapshots_path
        self.test_result_file = args.test_result_file
        self.tepoch = args.tepoch



    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        if self.optimizer_name.lower() != 'adam':
            raise ValueError(f"Unknown Optimizer: {self.optimizer_name}")
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2)


        
    def write_results(self, model, args, corpus, snap_idx):
        v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
        t_results = Inference.Test(args, model, corpus, 'test', snap_idx)
        logging.info("Trained model testing")

        val_str = Inference.print_results(None, v_results, None)
        val_result_filename_ = os.path.join(self.test_result_file, 'val_snap{}.txt'.format(snap_idx))
        open(val_result_filename_, 'w+').write(val_str)
        test_str = Inference.print_results(None, None, t_results)
        result_filename_ = os.path.join(self.test_result_file, 'test_snap{}.txt'.format(snap_idx))
        open(result_filename_, 'w+').write(test_str)

        ### save the test results to a json file for later analysis ###
        Ks = [10,20,50,100]
        json_t = {f'{metric}@{k}': v for metric, v in zip(['Recall', 'NDCG', 'MRR', 'Precision'], t_results) for k, v in zip(Ks, v)}
        with open (os.path.join(self.test_result_file, 'test_snap{}.json'.format(snap_idx)), 'w') as f:
            json.dump(json_t, f, indent=4)

    def train(self,
              model,
              data_dict,
              args,
              corpus,
              prev_data,
              snap_idx, 
              force_train):
        
        logging.info('Training time stage: {}'.format(snap_idx))

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        
        if snap_idx == 0 and os.path.exists(model.model_path+'_snap{}'.format(0)) and force_train == False:
            logging.info('Time_idx {} model already exists. Skip training and test directly.'.format(snap_idx))
            # test the model
            model.load_model(model.model_path+'_snap{}'.format(snap_idx))
            self.write_results(model, args, corpus, snap_idx)
            return 0, 0
        
        ### assuming existing models are properly trained ###
        elif os.path.exists(model.model_path+'_snap{}'.format(snap_idx)) and force_train == False:
            logging.info('Time_idx {} model already exists. Skip training and test directly.'.format(snap_idx))
            model.load_model(model.model_path+'_snap{}'.format(snap_idx))
            self.write_results(model, args, corpus, snap_idx)
            self.write_results_for_different_user_group(model, args, corpus, snap_idx)

            return 0, 0
        
        else:
            # pretrain model with time=0
            if snap_idx > 0 and 'pretrain' in args.dyn_method:
                logging.info('Time_idx {} model already exists. Skip training and test directly.'.format(snap_idx))
                model.load_model(model.model_path+'_snap0')
                self.write_results(model, args, corpus, snap_idx)
                return 0, 0
            else:
                logging.info('Time_idx {} model does not exist. Start training.'.format(snap_idx))


        # load previous model
        if snap_idx > 0 and 'finetune' in args.dyn_method:
            #print(model.model_path+'_snap{}'.format(idx-1))
            model.load_model(model.model_path+'_snap{}'.format(snap_idx-1))

        self._check_time(start=True)
        self.time_d = {}
        logging.info('dyn_method: {}'.format(self.dyn_method))
        if 'finetune' in self.dyn_method or 'newtrain' in self.dyn_method:
            num_epoch = self.tepoch
            shuffle = True
            if snap_idx == 0:
                num_epoch = self.epoch
                shuffle = True
        elif 'fulltrain' in self.dyn_method or 'pretrain' in self.dyn_method:
            num_epoch = self.epoch
            shuffle = True

        cnt = 0
        best_recall = 0
        best_epoch = 0

        titer = tqdm(range(num_epoch), ncols=300)
        for epoch in titer:
            self._check_time()
            total_loss, flag = self.fit(model, data_dict,prev_data,snap_idx, shuffle)
            training_time = self._check_time()

            logging.info('Epoch {:<3} total_loss={:<.4f} [{:<.1f} s]'.format(
                            epoch + 1, total_loss, training_time))

            if flag:
                logging.info('NaN loss, stop training')
                break


            if 'finetune' in self.dyn_method or 'newtrain' in self.dyn_method:
                early_stop = 0
                patience = 20
                minimum = 0
                if snap_idx == 0:
                    early_stop = 20
                    patience = 20
                    minimum = 0
            elif 'fulltrain' in self.dyn_method or 'pretrain' in self.dyn_method:
                early_stop = 20
                patience = 20
                minimum = 0
                
            a = 0
            b = 0

            #if epoch >= 20 and (epoch+1) % 5 == 0:
            if  (epoch) >= minimum and (epoch+1) % 2 == 0:
                v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
                Inference.print_results(None, v_results, None)

                if v_results[a][b] > best_recall:
                    best_epoch = epoch+1
                    best_recall = v_results[a][b] # top-10 or top-20
                    model.save_model(add_path='_snap{}'.format(snap_idx))

                # early stopping
                if epoch+1 > early_stop:
                    #recall_list.append((epoch, v_results[1][0]))
                    if v_results[a][b] < best_recall:
                        cnt += 1
                    else:
                        cnt = 0
                    if cnt >= patience:
                        break
        
        logging.info("End train and valid. Best validation epoch is {:03d}.".format(best_epoch))
        model.load_model(model.model_path+'_snap{}'.format(snap_idx))
        self.write_results(model, args, corpus, snap_idx)

        return self.time[1] - self.time[0], best_epoch


    def fit(self, model, data, prev_data, snap_idx, shuffle):
        gc.collect()
        torch.cuda.empty_cache()

        dl = DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, num_workers=8, pin_memory=self.pin_memory)
        
        flag = 0
        for current in dl:
            current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current['batch_size'] = len(current['user_id'])
            prediction, total_loss = self.train_recommender_vanilla(dl,model, current, prev_data,snap_idx)
            flag = np.isnan(prediction).any()
            if flag: 
                break
           

        return np.mean(total_loss).item(), flag

    def train_recommender_vanilla(self, data, model, current, prev_data,time_idx):
        # Train recommender
        model.train()
        # Get recommender's prediction and loss from the ``current'' data at t
        prediction = model(current['user_id'], current['item_id'])
        #u_ids, i_ids, prev_data, data, snap_idx,reduction='mean'
        #self, data, prev_data, snap_idx,reduction
        total_loss = model.loss(current, reduction='mean')
        # Update the recommender
        model.optimizer.zero_grad()
        total_loss.backward()
        model.optimizer.step()


        return  prediction.cpu().data.numpy(), total_loss.cpu().data.numpy()

