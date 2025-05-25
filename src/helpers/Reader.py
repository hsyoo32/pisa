# -*- coding: UTF-8 -*-

import os
import time
import pickle
import logging
import math
import torch
from random import randint
import pandas as pd
import numpy as np
import datetime
from utils import utils
import copy
import os,sys
os.chdir(sys.path[0]) 

class Reader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--suffix', type=str, default='pisa',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='Sep of csv file.')
        parser.add_argument('--train_ratio', type=float, default=0.8,
                            help='Ratio of the train dataset')
        parser.add_argument('--fname', type=str, default='freq',
                            help='Freq (> 20 records) or whole')
        parser.add_argument('--s_fname', type=str, default='',
                            help='Specific data folder name')
        parser.add_argument('--n_snapshots', type=int, default=10,
                            help='Number of test snapshots')
        parser.add_argument('--split_type', type=str, default='size',
                            help='Data split type')
        parser.add_argument('--val_ratio', type=float, default=0.5,
                            help='Ratio of the validation data to test data within each data block')


        return parser


    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.suffix = args.suffix
        self.dataset = args.dataset
        self.train_ratio = args.train_ratio
        self.batch_size = args.batch_size
        self.fname = args.fname
        self.s_fname = args.s_fname
        self.random_seed = args.random_seed
        self.n_snapshots = args.n_snapshots 
        self.split_type = args.split_type
        self.val_ratio = args.val_ratio # ratio to test data
        
        t0 = time.time()
        self._read_data()
        self.n_users, self.n_items = self.data_df['user_id'].max()+1, self.data_df['item_id'].max()+1
        self.dataset_size = len(self.data_df)
        self.n_batches = math.ceil(self.dataset_size/self.batch_size)
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(self.n_users, self.n_items, self.dataset_size))
        #self.path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname)
        self.path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self._set_snap_boundaries()
        self._save_snapshot_files()
        self.user_list = self.data_df['user_id'].to_numpy()
        self._save_user_clicked_set()

        del self.df

        logging.info('Done! [{:<.2f} s]'.format(time.time() - t0) + os.linesep)


    def _set_snap_boundaries(self):
        if 'size' in self.split_type:
            self.n_train_batches = int(self.dataset_size*self.train_ratio)
            self.n_test_batches = self.dataset_size - self.n_train_batches
            self.incre_size = int((self.dataset_size - self.n_train_batches) / self.n_snapshots)

            self.snap_boundaries = []
            for snapshot_idx in range(self.n_snapshots):
                self.snap_boundaries.append(self.n_train_batches + snapshot_idx * self.incre_size)

            tmp = copy.deepcopy(self.snap_boundaries)
            tmp.append(self.dataset_size)

        # elif 'time' in self.split_type:
        #     data = self.df.values.astype(np.int64)
        #     date = []
        #     for d in data:
        #         t = datetime.datetime.fromtimestamp(int(d[2])).timetuple()
        #         date.append([t[0],t[1]])

        #     # Split input data into snapshots divided by the pre-defined number of months
        #     prev_month = -1
        #     snapshots = {}

        #     k = -1
        #     threshold = threshold_cnt = self.n_snapshots # within how many months/years?
        #     for d in date:
        #         # month
        #         if d[1] == prev_month:
        #             snapshots[k].append(d)
        #         elif threshold_cnt < threshold:
        #             threshold_cnt += 1
        #             snapshots[k].append(d)
        #             prev_month = d[1]
        #         else:
        #             k += 1
        #             snapshots[k] = []
        #             snapshots[k].append(d)
        #             prev_month = d[1]
        #             threshold_cnt = 1

        #     accum_snapshots = {}
        #     for k, snap in snapshots.items():
        #         # 0,1,..,n-1,n
        #         accum_snapshots[k] = []
        #         for i in range(k+1):
        #             accum_snapshots[k].extend(snapshots[i])
                    
        #     snap_boundaries = []
        #     for k, snap in accum_snapshots.items():
        #         snap_boundaries.append(round(len(snap)/self.batch_size))

        #     # Here, self.train_ratio is the number of time periods for the training data
        #     print(self.train_ratio)
        #     self.n_train_batches = snap_boundaries[int(self.train_ratio)-1]
        #     self.n_test_batches = self.n_batches - self.n_train_batches

        #     print('ori_snap_boundaries: {}'.format(snap_boundaries))

        #     snap_boundaries = snap_boundaries[int(self.train_ratio):-1]
        #     for i, _ in enumerate(snap_boundaries):
        #         snap_boundaries[i] -= self.n_train_batches
        #     self.snap_boundaries = snap_boundaries

        #     print('snap_boundaries: {}'.format(self.snap_boundaries))


    def _save_snapshot_files(self):
        self.snapshots_path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname, 'snapshots')
        if not os.path.exists(self.snapshots_path):
            os.mkdir(self.snapshots_path)

        for idx, snap_boundary in enumerate(self.snap_boundaries):
            snapshot_train = self.data_df[:(snap_boundary)].values.astype(np.int64)
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'hist_block'+str(idx)), snapshot_train)

            if idx == 0:
                finetune_train = snapshot_train
            else:
                gap = self.snap_boundaries[idx] - self.snap_boundaries[idx-1]
                finetune_train = self.data_df[(snap_boundary - gap):(snap_boundary)].values.astype(np.int64)
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'incre_block'+str(idx)), finetune_train)

            # validation and test set

            if idx == len(self.snap_boundaries)-1:
                gap = self.dataset_size - (snap_boundary)
                gap = int(gap * self.val_ratio)
                val_block = self.data_df[(snap_boundary):(snap_boundary) + gap].values.astype(np.int64)
                test_block = self.data_df[(snap_boundary) + gap:].values.astype(np.int64)
            else:
                gap = (self.snap_boundaries[idx+1] - self.snap_boundaries[idx]) 
                gap = int(gap * self.val_ratio)
                val_block = self.data_df[(snap_boundary):(snap_boundary) + gap].values.astype(np.int64)
                test_block = self.data_df[(snap_boundary) + gap:(self.snap_boundaries[idx+1])].values.astype(np.int64)

            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'val_block'+str(idx)), val_block)
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, 'test_block'+str(idx)), test_block)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\", suffix = \"{}\", fname = \"{}\" '.format(self.prefix, self.dataset, self.suffix, self.fname))
        self.df = pd.read_csv(os.path.join(self.prefix, self.dataset, self.suffix, self.fname +'.csv'), sep=self.sep)  # Let the main runner decide the ratio of train/test
        self.data_df = self.df.loc[:, ['user_id', 'item_id']]  #.values.astype(np.int64) # (number of items, 2)
        
    def _save_user_clicked_set(self):
        user_clicked_set_path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname, 'user_clicked_set.txt')
        logging.info('Load user_clicked_set')
        self.user_clicked_set = self.data_df.groupby(['user_id'])['item_id'].unique().to_dict()

        pickle.dump(self.user_clicked_set, open(user_clicked_set_path, 'wb'))
        logging.info('Saved user_clicked_set')

    def _randint_w_exclude(self, clicked_set):
        randItem = randint(1, self.n_items-1)
        return self._randint_w_exclude(clicked_set) if randItem in clicked_set else randItem