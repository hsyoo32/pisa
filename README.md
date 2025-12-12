# PISA
**Balancing Stability and Plasticity in Continual Recommender Systems**  
SIGIR ’25

This repository implements PISA as proposed in the paper “Embracing Plasticity: Balancing Stability and Plasticity in Continual Recommender Systems” (SIGIR’25): link

---

## Requirements

* Python ≥ 3.7
* PyTorch ≥ 1.13.0
* scikit-learn
* tqdm
___

## Quickstart: Two-Stage Run

1. **Pre-train** a vanilla LGN on pre-training data

   ```bash
   python -u main.py \
     --model_name LGN \
     --dyn_method pretrain \
     --dataset Amazon-games \
     --epoch 500 \
     --tepoch 500 \
     --num_neg 4 \
     --train_ratio 0.6 \
     --lr 0.001 \
     --l2 1e-05 \
     --batch_size 1024 \
     --n_snapshots 5 \
     --split_type size \
     --random_seed 2021 \
     --gpu 0
   ```

2. **Run PISA** on the pre-trained model

   ```bash
   python -u main.py \
     --model_name PISA_LGN \
     --dyn_method finetune-plasticity-stability-userneigh \
     --dataset Amazon-games \
     --epoch 500 \
     --tepoch 500 \
     --num_neg 4 \
     --train_ratio 0.6 \
     --lr 0.001 \
     --l2 1e-05 \
     --batch_size 1024 \
     --n_snapshots 5 \
     --split_type size \
     --random_seed 2021 \
     --bound_weight 0.5 \
     --ratio 0.2 \
     --gpu 0
   ```

3. **(Added) Run Contrastive Learning Methods** (LWC-KD or LWC-KD-PIW)

   Files added in the these folders:
   - `Runner_cons.py` → `src/helpers/` (same folder as `Runner.py`)
   - `Contrastive.py` → `src/models/` (same folder as `Model.py`)
   - `Contrastive_LGN.py` → `src/models/general/` (same folder as `LGN.py`)

   Then run with the contrastive methods:

   ```bash
   python -u main.py \
     --model_name Contrastive_LGN \
     --dyn_method finetune-contra-onlyuser \
     --dataset Amazon-games \
     --epoch 500 \
     --tepoch 500 \
     --num_neg 4 \
     --train_ratio 0.6 \
     --lr 0.001 \
     --l2 1e-05 \
     --batch_size 1024 \
     --n_snapshots 5 \
     --split_type size \
     --random_seed 2021 \
     --bound_weight 0.5 \
     --gpu 0
   ```

---


## Command-Line Options

* `--dataset <name>`
  Choose from `Amazon-cds`, `Amazon-games`, or `Gowalla`.

* `--model_name <LGN|PISA_LGN>`
  `LGN` for vanilla backbone, or `PISA_LGN` for our stability-plasticity method.

* `--dyn_method <strategy>`

  * For **LGN**:

    * `pretrain`
    * `finetune`
    * `fulltrain`
  * For **PISA\_LGN**:

    * `finetune-plasticity-userneigh` (plasticity only)
    * `finetune-stability-userneigh` (stability only)
    * `finetune-plasticity-stability-userneigh` (full PISA)
      Additionally requires:

      * `--bound_weight <float>`: α – weight of the plasticity/stability enhancement loss.
      * `--ratio <float>`: L – top/bottom-L% selection ratio for personalized weighting.

  * For **Contrastive\_LGN** (Contrastive Learning Methods):

    * `finetune-contra-onlyuser` (LWC-KD)
    * `finetune-contra-piw-stdev-reinit-onlyuser` (LWC-KD-PIW)
      Additionally requires:

      * `--bound_weight <float>`: weight for the additional contrastive loss term (e.g., 0.5, 1.0, ...).

* `--n_snapshots <int>`
  Number of incremental data blocks.

* `--split_type <size>`
  How to partition the timeline (e.g., by interaction count).

* `--train_ratio <float>`
  Fraction of data used in the initial (pretrain) phase.

* `--num_neg <int>`
  Number of negative samples per positive.

* `--epoch <int>`
  Max epochs for the main training phase.

* `--tepoch <int>`
  Max epochs for each incremental update.

* `--lr <float>`
  Learning rate.

* `--l2` / `--decay <float>`
  L2 regularization weight.

* `--batch_size <int>`
  Mini-batch size.

* `--random_seed <int>`
  Seed for reproducibility.

* `--gpu <int>`
  GPU device ID.

---

## Batch Sweeps

Use our `_tester.py` script to sweep over seeds, α, and L:

```bash
python _tester.py
```

Edit the constants at the top of `_tester.py` to adjust datasets, models, strategies, and hyperparameter ranges.
