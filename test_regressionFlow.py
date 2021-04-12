import logging

import numpy as np
import torch
import torch.optim as optim

import backbone
import configs
from io_utils import parse_args_regressionFlow
from methods.DKT_regressionFlow import DKT
from methods.feature_transfer_regression import FeatureTransfer

from train_misc import set_cnf_options
from train_misc import add_spectral_norm
from train_misc import create_regularization_fns
from train_misc import build_model_tabular, build_conditional_cnf

# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info('Device: {}'.format(device))

params = parse_args_regressionFlow('test_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
bb = backbone.Conv3().to(device)
if params.use_conditional:
    cnf = build_conditional_cnf(params, 1, params.context_dim).to(device)
else:
    regularization_fns, regularization_coeffs = create_regularization_fns(params)
    cnf = build_model_tabular(params, 1, regularization_fns).to(device)
if params.spectral_norm: add_spectral_norm(cnf)
set_cnf_options(params, cnf)

if params.method == 'DKT':
    model = DKT(bb, cnf, device, params.use_conditional)
    optimizer = None
elif params.method == 'transfer':
    model = FeatureTransfer(bb, device)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
else:
    ValueError('Unrecognised method')

model.load_checkpoint(params.checkpoint_dir)

mse_list = []
nll_list = []
for epoch in range(params.n_test_epochs):
    mse, NLL = model.test_loop(params.n_support, optimizer)
    mse = float(mse.cpu().detach().numpy())
    NLL = float(NLL.cpu().detach().numpy())
    mse_list.append(mse)
    nll_list.append(NLL)

print("-------------------")
print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
print("-------------------")
print("-------------------")
print("Average NLL: " + str(np.mean(nll_list)) + " +- " + str(np.std(nll_list)))
print("-------------------")