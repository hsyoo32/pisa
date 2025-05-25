# -*- coding: UTF-8 -*-

import os
import shutil
import sys
import pickle
import logging
import argparse
import torch

from helpers import Reader, Runner, Runner_PISA
from models import  Model
from models.general import LGN, PISA_LGN
from models import Dataloader
from utils import utils
import pandas as pd

def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--result_file', type=str, default='',
                        help='Result file path')
    # parser.add_argument('--result_folder', type=str, default='',
    #                     help='Result folder path')
    parser.add_argument('--random_seed', type=int, default=2021,
                        help='Random seed of numpy and pytorch.')
    # parser.add_argument('--train', type=int, default=1,
    #                     help='To train the model or not.')
    # parser.add_argument('--regenerate', type=int, default=0,
    #                     help='Whether to regenerate intermediate files.')
    parser.add_argument('--dyn_update', type=int, default=0,
                        help='dynamic update strategy.')
    parser.add_argument('--t_opt', type=int, default=5,)
    return parser

def test_file(args, corpus, test_type):
    d = {}
    for idx in range(corpus.n_snapshots):
        val_result_filename_ = os.path.join(args.test_result_file, '{}_snap{}.txt'.format(test_type, idx))
        with open(val_result_filename_, 'r') as f:
            lines = f.readlines()
            data = [line.replace('\n', '').split() for line in lines]
            for value in data:
                if d.get(value[0]) is None:
                    d[value[0]] = []
                d[value[0]].append(float(value[1]))

    with open(os.path.join(args.test_result_file, '_{}_mean'.format(test_type)), 'w+') as f:

        for k, v in d.items():
            # skip the first value
            v = v[1:]
            f.writelines('{}\t{:.4f}\n'.format(k, sum(v)/len(v)))
    
    with open(os.path.join(args.test_result_file, '_{}_trend'.format(test_type)), 'w+') as f:
        for k, v in d.items():
            f.writelines('{}'.format(k))
            for v_ in v:
                f.writelines('\t{:.4f}'.format(v_))
            f.writelines('\n')

def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)

    # Random seed
    utils.fix_seed(args.random_seed)

    # GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('cuda available: {}'.format(torch.cuda.is_available()))
    logging.info('cuda device: {}'.format(args.gpu))

    # Read data
    corpus_path = os.path.join(args.path, args.dataset, args.suffix, args.s_fname, model_name.reader + '.pkl')


    if os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pd.read_pickle(corpus_path)
    else:
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pd.to_pickle(corpus, corpus_path)

    args.keys = ['train', 'test']
    logging.info('Total instances: {}'.format(corpus.dataset_size))
    logging.info('Train instances: {}'.format(corpus.n_train_batches))
    logging.info('Test instances: {}'.format(corpus.n_test_batches))
    logging.info('Snap boundaries: {}'.format(corpus.snap_boundaries + [corpus.dataset_size]))

    # Run model 
    runner = runner_name(args, corpus)
    data_dict = dict()

    # full-retraining   
    #utils.fix_seed(args.random_seed)

    if 'fulltrain' in args.dyn_method or 'pretrain' in args.dyn_method :
        data_type = 'hist'
    elif 'finetune' in args.dyn_method or 'newtrain' in args.dyn_method :
        data_type = 'incre'
        
    print('Data type: ', data_type)    
    time_d = {}
    best_epoch_d = {}
    prev_data=[]

    force_train = False

    for idx in range(corpus.n_snapshots):
        data_dict = Dataloader.Dataset(args, corpus, data_type, idx)

        utils.fix_seed(args.random_seed)
        model = model_name(args, corpus, data_dict, idx)
        model.apply(model.init_weights)
        model.to(model._device)
        
        # if idx == 0 and pretrained == True and os.path.exists(args.model_path+'_snap{}'.format(0)):
        #     print('Time idx 0 pretrained model exist. skip both pretraining and test')
        #     continue
        
        print('pretrain model path:',args.pretrain_model_path+'_snap0')
        #if idx == 0 and not os.path.exists(args.model_path+'_snap0') and os.path.exists(args.pretrain_model_path+'_snap0') and force_train == False:
        if idx == 0 and os.path.exists(args.pretrain_model_path+'_snap0') and force_train == False:
            print('Time idx 0 pretrained model exist. Copy the pretrained model to the model path')
            # copy the pretrained model to the model path
            # how to copy the file
            # if the model_path is pretrain_model_path, pass
            if args.model_path == args.pretrain_model_path:
                print('model_path is pretrain_model_path, pass')
            else:
                shutil.copy(args.pretrain_model_path+'_snap0', args.model_path+'_snap0')


        if idx > 0:
            prev_data = Dataloader.Dataset(args, corpus, 'hist', idx-1)

        if 'plasticity' in args.dyn_method:
            # Pure fine-tuning
            _ = runner.train(model, data_dict, args, corpus, prev_data, idx, force_train, 0)
            # PISA
            best_epoch = runner.train(model, data_dict, args, corpus, prev_data, idx, force_train, 1)
        else:
            best_epoch = runner.train(model, data_dict, args, corpus, prev_data, idx, force_train)
        #time_d['period_{}'.format(idx)] = t
        best_epoch_d['period_{}'.format(idx)] = best_epoch
        #best_epoch_list.append(best_epoch)


    # If there is keys/data in time_d, save it to file
    # if time_d:
    #     df = pd.DataFrame(list(time_d.items()), columns=['Period', 'Time'])
    #     df['Time (s)'] = df['Time']
    #     df['Time (m)'] = df['Time'] / 60
    #     # drop the 'Time' column
    #     df = df.drop(columns=['Time'])
    #     df.to_csv(args.test_result_file + '_time_test.csv', index=False)

    # If there is keys/data in best_epoch_d, save it to file 
    if best_epoch_d:
        df = pd.DataFrame(list(best_epoch_d.items()), columns=['Period', 'Best_epoch'])
        df.to_csv(args.test_result_file + '_best_epoch_test.csv', index=False)

    # mean and trend files (optional)
    test_file(args, corpus, 'test')
    #test_file(args, corpus, 'val')


    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)

def post():
    return args.test_result_file

def get_pretrain_model_path(args, log_args1, log_args2, params):
    log_args3 = []
    for arg in params:
        # for "tepoch", set to -1
        if arg == 'tepoch':
            log_args3.append(arg + '=' + str(-1))
        else:
            log_args3.append(arg + '=' + str(eval('args.' + arg)))

    log_file_name1 = '__'.join(log_args1).replace(' ', '__')
    log_file_name2 = '__'.join(log_args2).replace(' ', '__')
    log_file_name3 = '__'.join(log_args3).replace(' ', '__')

    model_path = '../model/{}/{}/{}/'.format(log_file_name1, log_file_name2, log_file_name3)
    utils.check_dir(model_path)
    return model_path


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BPR', help='Choose a model to run.')
    init_parser.add_argument('--dyn_method', type=str, default='pretrain', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    #print(init_args.model_name)
    model_name = eval('{0}.{0}'.format(init_args.model_name))
    reader_name = eval('{0}.{0}'.format(model_name.reader))
    runner_name = eval('{0}.{0}'.format(model_name.runner))
    #tester_name = eval('{}.{}'.format('Runner','Tester'))
    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    #parser = tester_name.parse_tester_args(parser)
    args, extras = parser.parse_known_args()

    if init_args.dyn_method == 'finetune':
        pass
    elif 'fulltrain' in init_args.dyn_method:
        args.tepoch = -1
    elif 'pretrain' in init_args.dyn_method:
        args.tepoch = -1


    args.s_fname = '{}_{}_{}_s{}_r{}'.format(args.split_type, args.train_ratio, args.batch_size, args.n_snapshots, args.t_opt)
    
    log_args1 = [args.dataset, args.s_fname]
    log_args2 = [init_args.model_name, init_args.dyn_method]
    log_args3 = []


    params = ['lr','l2','epoch','tepoch','num_neg','random_seed']

    args.pretrain_model_path = get_pretrain_model_path(args, log_args1, ['LGN', 'pretrain'], params)

    if 'PISA' in init_args.model_name:
        params = ['lr','l2','epoch','tepoch','num_neg','random_seed','bound_weight','ratio'] ###

    for arg in params:
        log_args3.append(arg + '=' + str(eval('args.' + arg)))

    log_file_name1 = '__'.join(log_args1).replace(' ', '__')
    log_file_name2 = '__'.join(log_args2).replace(' ', '__')
    log_file_name3 = '__'.join(log_args3).replace(' ', '__')
    ### for test

    if args.model_path == '':
        args.model_path = '../model/{}/{}/{}/'.format(log_file_name1, log_file_name2, log_file_name3)
    utils.check_dir(args.model_path)

    if args.log_file == '':
        args.log_file = '../log/{}/{}/{}.txt'.format(log_file_name1, log_file_name2, log_file_name3)
    utils.check_dir(args.log_file)

    if args.test_result_file == '':
        args.test_result_file = '../test_result/{}/{}/{}/'.format(log_file_name1, log_file_name2, log_file_name3)
    utils.check_dir(args.test_result_file)
    
    args.dyn_method = init_args.dyn_method
    args.model_name = init_args.model_name
    
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(log_file_name1+'__'+log_file_name2)
    main()