Initial implementation of 

#### "Non-Gaussian Gaussian Processes for Few-Shot Regression"

**final version will be released to github after the final decisions of NeurIPS**

## Overview

Gaussian Processes (GPs) have been widely used in machine learning to model distributions over functions, with applications including multi-modal regression, time-series prediction, and few-shot learning. GPs are particularly useful in the last application
since they rely on Normal distributions and, hence, enable closed-form computation of the posterior probability function.
Unfortunately, because the resulting posterior is not flexible enough to capture complex distributions, GPs assume high similarity between subsequent tasks -- a~requirement rarely met in real-world conditions.
In this work, we address this limitation by leveraging the flexibility of Normalizing Flows to modulate the posterior predictive distribution of the GP. This makes the GP posterior locally non-Gaussian, therefore we name our method Non-Gaussian Gaussian Processes (NGGPs). 
We propose an invertible ODE-based mapping that operates on each component of the random variable vectors and shares the parameters across all of them. 
We empirically tested the flexibility of NGGPs on various few-shot learning regression datasets, showing that the mapping can incorporate context embedding information to model different noise levels for periodic functions.
As a result, our method shares the structure of the problem between subsequent tasks, but the contextualization allows for adaptation to dissimilarities.
NGGPs outperform the competing state-of-the-art approaches on a diversified set of benchmarks and applications.

## Requirements
All necessary libraries are in `environment.yml`.
 

## Experiments
Exemplary `DKT` usage:
```
source activate object_tracking

python run_regression.py \
--dataset QMUL \
--model=Conv3 \
--method="DKT" \
--output_dim=1 \
--seed=1 \
--save_dir ./save/"regression_DKT_QMUL" \
--kernel_type rbf \
--stop_epoch 1000
```

Exemplary `NGGP` usage:
```
source activate object_tracking

python run_regression.py \
--dataset sines \
--model=MLP2 \
--method="DKT" \
--output_dim=40 \
--seed=1 \
--save_dir ./save/flow_sines_rbf" \
--kernel_type rbf \
--stop_epoch 50000 \
--all_lr 1e-3 \
--meta_batch_size 1 \
--multidimensional_phase False \
--multidimensional_amp False \
--noise hetero_multi \
--use_conditional True \
--context_type backbone \
--n_test_epochs 500 \
--n_support 5 \
--flow
```

This will run the training on the sines dataset with heteroskedastic noise. In order to change the QMUL datasets provide `--dataset QMUL`. In order to perform the test add to the above command line options `--test`. See the *Supplementary Materials A* for the description of the used in the paper hyperparameters for each dataset. 

## Acknowledgements

This repository is a fork of: [https://github.com/BayesWatch/deep-kernel-transfer](https://github.com/BayesWatch/deep-kernel-transfer)
