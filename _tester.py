import os
import itertools

# --- Data & Models ---
datasets    = ['Amazon-games','Amazon-cds']  # Options: 'Amazon-cds', 'Amazon-games', 'Gowalla'
emb_algos   = ['LGN']  # Options: 'LGN', 'PISA_LGN'

# Vanilla strategies for LGN
'''vanilla_strats = ['pretrain', 'finetune', 'fulltrain']'''
vanilla_strats = ['pretrain']
# PISA strategies for PISA_LGN
'''pisa_strats = [
    'finetune-plasticity-userneigh',          # plasticity only
    'finetune-stability-userneigh',           # stability only
    'finetune-plasticity-stability-userneigh' # both (full PISA)
]'''
pisa_strats = ['finetune-plasticity-stability-userneigh']

num_neg       = 4         
n_snapshots   = 5         
epochs        = 500       
tepochs       = 500       
lr            = 0.001     
l2            = '1e-05'   
batch_size_def= 1024
train_ratio   = 0.6
gpu           = 3

# --- Sweep parameters ---
bound_weights = [0.5]     # α in paper
ratios        = [1.0,0.2]     # L in paper
seeds         = [2021]    # [2021, 2022, 2023, 2024, 2025]

os.chdir('src')
for data, model in itertools.product(datasets, emb_algos):
    strats = vanilla_strats if model == 'LGN' else pisa_strats

    for strat in strats:
        for seed in seeds:
            # PISA_LGN uses α & L; LGN uses zeros
            bounds = bound_weights if model == 'PISA_LGN' else [0.0]
            rts    = ratios        if model == 'PISA_LGN' else [0.0]
            bs     = 256 if 'Gowalla' in data else batch_size_def

            for bound, ratio in itertools.product(bounds, rts):
                cmd = (
                    f"python main.py "
                    f"--model_name {model} --gpu {gpu} "
                    f"--epoch {epochs} --tepoch {tepochs} "
                    f"--dataset {data} --num_neg {num_neg} "
                    f"--dyn_method {strat} --train_ratio {train_ratio} "
                    f"--lr {lr} --l2 {l2} --batch_size {bs} "
                    f"--n_snapshots {n_snapshots} --split_type size "
                    f"--random_seed {seed} --bound_weight {bound} --ratio {ratio}"
                )
                print(cmd)
                os.system(cmd)