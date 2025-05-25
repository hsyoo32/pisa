import os
import gc
import copy
import torch
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Dataloader
from utils import utils
from models.Model import Model
import Inference
import json

class Runner_PISA:
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100, help='Number of epochs.')
        parser.add_argument('--tepoch', type=int, default=200, help='Number of epochs for fine-tuning.')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--l2', type=float, default=1e-04, help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size during training.')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer: Adam')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of processors for DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1, help='pin_memory in DataLoader')
        parser.add_argument('--test_result_file', type=str, default='', help='Path for test results')
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
        self.test_result_file = args.test_result_file
        self.tepoch = args.tepoch
        self.time = None
        self.snap_boundaries = corpus.snap_boundaries
        self.snapshots_path = corpus.snapshots_path

    def _build_optimizer(self, model):
        if self.optimizer_name.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
        raise ValueError(f"Unknown Optimizer: {self.optimizer_name}")

    def write_results(self, model, args, corpus, snap_idx, option=''):
        """Write validation and test results to files and save metrics in JSON format."""
        v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
        t_results = Inference.Test(args, model, corpus, 'test', snap_idx)
        logging.info("Trained model testing")

        # Save validation results
        val_str = Inference.print_results(None, v_results, None)
        val_path = os.path.join(self.test_result_file, f'{option}val_snap{snap_idx}.txt')
        open(val_path, 'w+').write(val_str)

        # Save test results
        test_str = Inference.print_results(None, None, t_results)
        test_path = os.path.join(self.test_result_file, f'{option}test_snap{snap_idx}.txt')
        open(test_path, 'w+').write(test_str)

        # Save metrics to JSON
        Ks = [10, 20, 50, 100]
        metrics = ['Recall', 'NDCG', 'MRR', 'Precision']
        json_results = {f'{metric}@{k}': v for metric, values in zip(metrics, t_results) for k, v in zip(Ks, values)}
        json_path = os.path.join(self.test_result_file, f'{option}test_snap{snap_idx}.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)

    def train(self, model, data_dict, args, corpus, prev_data, snap_idx, force_train=False, step_flag=0):
        """Main training loop with early stopping and model checkpointing."""
        logging.info(f'Training time stage: {snap_idx}')

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        # Check if model exists and handle accordingly
        model_path = f'{model.model_path}_snap{snap_idx}'
        if os.path.exists(model_path) and not force_train:
            if not (step_flag == 0 and 'plasticity' in self.dyn_method):
                model.load_model(model_path)
                self.write_results(model, args, corpus, snap_idx)
                print(f'model already exists, skip training')
            return 0, 0
        else:
            print(f'model does not exist, training from scratch for time stage {snap_idx}')

        # Handle pretraining case
        # if snap_idx > 0 and 'pretrain' in args.dyn_method:
        #     model.load_model(f'{model.model_path}_snap0')
        #     self.write_results(model, args, corpus, snap_idx)
        #     return 0, 0

        # Load previous model for fine-tuning
        prev_model = None
        forward_model = None
        if snap_idx > 0 and 'finetune' in args.dyn_method:
            model.load_model(f'{model.model_path}_snap{snap_idx-1}')
            model.freeze_flag = 0
            prev_model = copy.deepcopy(model)
            prev_model.eval()

            if step_flag > 0:
                forward_model = copy.deepcopy(model)
                forward_model.load_model(f'{model.model_path}_forward_snap{snap_idx}')
                forward_model.eval()

        # Training loop
        num_epoch = self.tepoch if ('finetune' in self.dyn_method or 'newtrain' in self.dyn_method) else self.epoch
        if snap_idx == 0:
            num_epoch = self.epoch

        best_recall = 0
        best_epoch = 0
        patience = 20
        cnt = 0
        model.forward_flag = step_flag

        for epoch in tqdm(range(num_epoch), ncols=100, mininterval=1):
            model.epoch = epoch
            losses = self.fit(model, data_dict, prev_data, snap_idx, True, prev_model, forward_model)
            logging.info(f'Epoch {epoch} total_loss={losses[0]:.4f} bpr_loss={losses[1]:.4f} cl_loss={losses[2]:.4f} plast_loss={losses[3]:.4f} stab_loss={losses[4]:.4f} plast_neigh_loss={losses[5]:.4f} stab_neigh_loss={losses[6]:.4f}')

            if np.isnan(losses[0]).any():
                logging.info('NaN loss, stop training')
                exit()

            # Validation and early stopping
            if epoch >= 0 and (epoch + 1) % 2 == 0:
                v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
                if v_results[0][0] > best_recall:
                    best_epoch = epoch + 1
                    best_recall = v_results[0][0]
                    save_path = f'_forward_snap{snap_idx}' if (step_flag == 0 and 'plasticity' in self.dyn_method) else f'_snap{snap_idx}'
                    model.save_model(add_path=save_path)

                if epoch + 1 > 20:
                    if v_results[0][0] < best_recall:
                        cnt += 1
                    else:
                        cnt = 0
                    if cnt >= patience:
                        break

        logging.info(f"Training complete. Best validation epoch: {best_epoch:03d}")
        
        # Load best model and write results
        model_path = f'{model.model_path}_forward_snap{snap_idx}' if (step_flag == 0 and 'plasticity' in self.dyn_method) else f'{model.model_path}_snap{snap_idx}'
        model.load_model(model_path)
        self.write_results(model, args, corpus, snap_idx, option='forward' if step_flag == 0 and 'plasticity' in self.dyn_method else '')

        return best_epoch

    def fit(self, model, data, prev_data, snap_idx, shuffle, prev_model, forward_model):
        """Single epoch training loop with k-means updates and batch processing."""
        with torch.no_grad():
            if not ('plasticity' in self.dyn_method and model.forward_flag == 0):
                model.update_kmeans(prev_model)

        gc.collect()
        torch.cuda.empty_cache()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, 
                       num_workers=self.num_workers, pin_memory=self.pin_memory)

        total_losses = []
        for current in dl:
            current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current['batch_size'] = len(current['user_id'])
            losses = self.train_recommender_vanilla(data, model, current, prev_data, snap_idx, prev_model, forward_model)
            total_losses.append(losses)

        return [np.mean(loss).item() for loss in zip(*total_losses)]

    def train_recommender_vanilla(self, data, model, current, prev_data, time_idx, prev_model, forward_model):
        """Process a single batch of data and update model parameters."""
        model.train()
        losses = model.loss(data, current, prev_data, time_idx, prev_model, forward_model, reduction='mean')
        
        model.optimizer.zero_grad()
        losses[0].backward()
        model.optimizer.step()

        return [loss.cpu().data.numpy() for loss in losses]