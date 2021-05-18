#!/bin/bash

#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=dgx
#SBATCH --cpus-per-gpu=4


source activate object_tracking

python run_regression.py \
--neptune \
--dataset sines \
--model=MLP2 \
--method="DKT" \
--output_dim=40 \
--seed=1 \
--save_dir ./save/"sines1d_hetero3/flow_$1" \
--kernel_type $1 \
--stop_epoch 5000 \
--all_lr 1e-3 \
--meta_batch_size 1 \
--multidimensional_phase False \
--multidimensional_amp False \
--noise heterogeneous \
--use_conditional True \
--context_type backbone \
--n_test_epochs 5 \
--n_support 5 \
--flow

