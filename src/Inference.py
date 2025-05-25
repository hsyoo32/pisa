import torch
import numpy as np
import os
from utils import utils
import math
import logging

def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = []
    #归一化折扣累计增益
    NDCG = []
    #平均倒数排名
    MRR = []
    #user_list = [1309,1662,1961] # stable
    #user_list = [1941,1909,1799,1662,1626,1624,1600] # unstable
    for index in range(len(topN)):
        # print(f'top {topN[index]}\n')
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                #是否计算mrr
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                #理想的折扣累计增益
                idcg = 0
                #剩余的理想命中计数
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                cnt += 1
            # else: 
            #     print('OPS')
#             if i in user_list:
#                 print(f'user {i}')
# #                 print(predictedIndices[i])
# #                 print(GroundTruth[i])
#                 print(userHit / len(GroundTruth[i]))
#                 print(ndcg)
        precision.append(round(sumForPrecision / cnt, 4))
        recall.append(round(sumForRecall / cnt, 4))
        NDCG.append(round(sumForNdcg / cnt, 4))
        MRR.append(round(sumForMRR / cnt, 4))
        
    return recall, NDCG, MRR, precision


def Test_group(args, model, corpus, data_type, data_idx, group_files):
    """
    Args:
        group_files: Dictionary containing paths to user group files
            e.g., {"dynamic": "dynamic_users.txt", "static": "static_users.txt", "intermediate": "intermediate_users.txt"}
    """

    batch_size = args.batch_size
    model.eval()

    # Load user groups from files
    user_groups = {}
    for group_name, file_path in group_files.items():
        with open(file_path, "r") as f:
            user_groups[group_name] = set(int(line.strip()) for line in f)

    test_file = os.path.join(corpus.snapshots_path, data_type + '_block' + str(data_idx))
    test_data = utils.read_data_from_file_int(test_file)
    test_data = np.array(test_data)
    test_pos_items = {}
    for user, item in test_data:
        if user not in test_pos_items:
            test_pos_items[user] = []
        test_pos_items[user].append(item)

    hist_file = os.path.join(corpus.snapshots_path, 'hist_block' + str(data_idx))
    hist_data = utils.read_data_from_file_int(hist_file)
    hist_data = np.array(hist_data)
    hist_pos_items = {}
    for user, item in hist_data:
        if user not in hist_pos_items:
            hist_pos_items[user] = []
        hist_pos_items[user].append(item)

    Ks = [10, 20, 50, 100]
    max_K = max(Ks)

    group_results = {}

    with torch.no_grad():
        # Evaluate each user group
        for group_name, group_users in user_groups.items():
            users = [u for u in test_pos_items.keys() if u in group_users]
            if not users:
                continue

            users_list = []
            rating_list = []
            ground_truth_list = []

            n_batch = len(users) // batch_size
            if len(users) % batch_size != 0:
                n_batch += 1

            for i in range(n_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(users))
                batch_users = users[start:end]

                all_pos = []
                for user in batch_users:
                    # Cold-start users
                    if user not in hist_pos_items:
                        all_pos.append([])
                    else:
                        all_pos.append(hist_pos_items[user])

                ground_truth = []
                for user in batch_users:
                    ground_truth.append(test_pos_items[user])

                user_id = torch.tensor(batch_users, dtype=torch.int64).to(model._device)
                item_id = torch.tensor(np.arange(corpus.n_items), dtype=torch.int64).to(model._device)

                scores = model.infer_user_scores(user_id, item_id)
                scores = scores.cpu().numpy()

                exclude_index = []
                exclude_items = []
                for i, items in enumerate(all_pos):
                    exclude_index.extend([i] * len(items))
                    exclude_items.extend(items)
                scores[exclude_index, exclude_items] = -np.inf

                _, rating_K = torch.topk(torch.tensor(scores), k=max_K)

                users_list.append(batch_users)
                rating_list.extend(rating_K.cpu())
                ground_truth_list.extend(ground_truth)

            recall, NDCG, MRR, precision = computeTopNAccuracy(ground_truth_list, rating_list, Ks)
            group_results[group_name] = (recall, NDCG, MRR, precision)

    return group_results


def Test(args, model, corpus, data_type, data_idx):
    batch_size = args.batch_size
    model.eval()

    #data_type = 'val'

    #dataset = Dataloader.Dataset(model, args, corpus, data_type, idx)
    test_file = os.path.join(corpus.snapshots_path, data_type+'_block'+str(data_idx))
    test_data = utils.read_data_from_file_int(test_file)
    test_data = np.array(test_data)
    test_data = test_data
    test_pos_items = {}
    for user, item in test_data:
        if user not in test_pos_items:
            test_pos_items[user] = []
        test_pos_items[user].append(item)

    hist_file = os.path.join(corpus.snapshots_path, 'hist_block'+str(data_idx))
    hist_data = utils.read_data_from_file_int(hist_file)
    
    hist_data = np.array(hist_data)
    hist_pos_items = {}
    for user, item in hist_data:
        if user not in hist_pos_items:
            hist_pos_items[user] = []
        hist_pos_items[user].append(item)

    Ks = [10,20,50,100]
    max_K = max(Ks)

    users_list = []
    rating_list = []
    ground_truth_list = []

    with torch.no_grad():
        users = list(test_pos_items.keys())
        n_batch = len(users) // batch_size
        if len(users) % batch_size != 0:
            n_batch += 1
        
        for i in range(n_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(users))
            batch_users = users[start:end]

            all_pos = []
            for user in batch_users:
                # cold-start users
                if user not in hist_pos_items:
                    all_pos.append([])
                else:
                    all_pos.append(hist_pos_items[user])
            
            ground_truth = []
            for user in batch_users:
                ground_truth.append(test_pos_items[user])

            user_id = torch.tensor(batch_users, dtype=torch.int64).to(model._device)
            item_id = torch.tensor(np.arange(corpus.n_items), dtype=torch.int64).to(model._device)

            scores = model.infer_user_scores(user_id, item_id)
            scores = scores.cpu().numpy()

            exclude_index = []
            exclude_items = []
            for i, items in enumerate(all_pos):
                exclude_index.extend([i] * len(items))
                exclude_items.extend(items)
            scores[exclude_index, exclude_items] = -np.inf

            _, rating_K = torch.topk(torch.tensor(scores), k=max_K)
            
            users_list.append(batch_users)
            rating_list.extend(rating_K.cpu())
            ground_truth_list.extend(ground_truth)
        
        assert n_batch == len(users_list)
        
        recall, NDCG, MRR, precision = computeTopNAccuracy(ground_truth_list, rating_list, Ks)

        # results = [recall, NDCG, MRR, precision]
        # string_results = {f'{metric}@{K}': v for metric, v in zip(['Recall', 'NDCG', 'MRR', 'Precision'], results) for K, v in zip(Ks, v)}

        return recall, NDCG, MRR, precision


def print_results(loss, valid_result, test_result):
    result_str = ''
    """output the evaluation results."""
    if loss is not None:
        logging.info("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        logging.info("[Valid]: Recall: {} NDCG: {} MRR: {} Precision: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
                # result_str += 'Top-10\n'
        result_str += 'Recall@10\t' + str(valid_result[0][0]) + '\n'
        result_str += 'NDCG@10\t' + str(valid_result[1][0]) + '\n'
        # result_str += 'Top-20\n'
        result_str += 'Recall@20\t' + str(valid_result[0][1]) + '\n'
        result_str += 'NDCG@20\t' + str(valid_result[1][1]) + '\n'
        # result_str += 'Top-50\n'
        result_str += 'Recall@50\t' + str(valid_result[0][2]) + '\n'
        result_str += 'NDCG@50\t' + str(valid_result[1][2]) + '\n'
    if test_result is not None: 
        logging.info("[Test]: Recall: {} NDCG: {} MRR: {} Precision: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))
        
        # result_str += 'Top-10\n'
        result_str += 'Recall@10\t' + str(test_result[0][0]) + '\n'
        result_str += 'NDCG@10\t' + str(test_result[1][0]) + '\n'
        # result_str += 'Top-20\n'
        result_str += 'Recall@20\t' + str(test_result[0][1]) + '\n'
        result_str += 'NDCG@20\t' + str(test_result[1][1]) + '\n'
        # result_str += 'Top-50\n'
        result_str += 'Recall@50\t' + str(test_result[0][2]) + '\n'
        result_str += 'NDCG@50\t' + str(test_result[1][2]) + '\n'

    return result_str