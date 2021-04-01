import logging
import os

import numpy as np
import torch

import backbone
import configs
from io_utils import parse_args_tracking
from methods.DKT_tracking import DKT, TrackingBackbone
from methods.feature_transfer_regression import FeatureTransfer
from data.data_regression_NGSIM import ngsimDataset

import matplotlib.pyplot as plt


def draw_ngsim_plots(hist, y_gt, y_pred, bid, dir):
    hist_np = hist.cpu().detach().numpy()
    y_gt = y_gt.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    plt.scatter(hist_np[:, 0, 1], hist_np[:, 0, 0], marker='^')
    #plt.scatter(y_pred[k, :, 1], y_pred[k, :, 0], marker='*', c='#ff7f0e')
    plt.scatter(y_pred[25:], y_pred[:25], marker='*', c='#ff7f0e')
    plt.scatter(y_gt[:, 0, 1], y_gt[:, 0, 0], marker='o', c='#bcbd22')
    plt.savefig(os.path.join(dir, str(bid) + '.png'))
    plt.close()

#torch.backends.cudnn.enabled = False

params = parse_args_tracking('train_regression')
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

params.checkpoint_dir = '%scheckpoints/%s/' % (configs.save_dir, params.dataset)
if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)
params.checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

save_dir = '%scheckpoints/%s/' % (configs.save_dir, params.dataset)

# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info('Device: {}'.format(device))

bb = TrackingBackbone().to(device)


if params.method == 'DKT':
    model = DKT(bb, device)

model.load_checkpoint(params.checkpoint_dir)

data = ngsimDataset(os.path.join(configs.data_dir['ngsim'], 'TestSet.mat'))
test_loader = torch.utils.data.DataLoader(data, batch_size=params.batch_size,
                                           shuffle=True, num_workers=8, collate_fn=data.collate_fn)

model.model.eval()
model.feature_extractor.eval()
mses = []
plotting = False
for k in range(5):
    mses.append([])
for bidx, data in enumerate(test_loader):
    if bidx < 10000:
        test_example, pred = model.test_loop(data)
        if test_example is not None:
            for k in range(5):
                mses[k].append(test_example[k])
            if plotting:
                hist, nbrs, mask, _, _, fut, op_mask = data
                draw_ngsim_plots(hist, fut, pred, bidx, save_dir)
    else:
        break

# 0.3048 - feets to meters
for k in range(5):
    print('MSE [%d s]: %.3f' %(k+1, sum(mses[k])/len(mses[k])*0.3048))