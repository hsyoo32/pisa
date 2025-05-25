# -*- coding: UTF-8 -*-
import os
import torch
import datetime
import numpy as np
import random


def read_data_from_file(filename, flag=0):
    with open(filename, "r") as f: 
        lines = f.readlines()
        if flag == 0:
            data = [line.replace('\n','').split() for line in lines]
        elif flag == 1:
            data = [line.replace('\n','').split('::') for line in lines]
    return data

def read_data_from_file_int(filename, flag=0):
    with open(filename, "r") as f: 
        lines = f.readlines()
        if flag == 0:
            data = [str_list_to_int(line.split()) for line in lines]
        elif flag == 1:
            data = [str_list_to_int(line.split('::')) for line in lines]
    return data

def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory - " + directory)

def get_user_dil_from_edgelist(edges):
    dil = {}
    for edge in edges:
        if dil.get(edge[0]) is None:
            dil[edge[0]] = []
        dil[edge[0]].append(edge[1])

    return dil

def get_user_item_set(edges):
    user_set = set()
    item_set = set()
    for edge in edges:
        user_set.add(edge[0])
        item_set.add(edge[1])
    return list(user_set), list(item_set)

def write_interactions_to_file(filename, data):

    with open(filename, 'w+') as f:
        for d in data:
            f.writelines('{}\t{}\n'.format(d[0],d[1]))

def batch_to_gpu(batch: dict, device) -> dict:
    for c in batch:
        if type(batch[c]) is torch.Tensor:
            batch[c] = batch[c].to(device)
    return batch

def squeeze_dict(batch: dict, dim=0) -> dict:
    for c in batch:
        if not torch.is_tensor(batch[c]):
            batch[c] = torch.from_numpy(batch[c])
        batch[c].squeeze_(dim)
    return batch

def check_dir(file_name: str):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


