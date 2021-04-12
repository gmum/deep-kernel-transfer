import logging
import os

import numpy as np
import torch

import backbone
import configs
from io_utils import parse_args_tracking
from methods.DKT_tracking_multivariate import DKT, TrackingBackbone
from methods.feature_transfer_regression import FeatureTransfer
from data.data_regression_NGSIM import ngsimDataset


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

# Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info('Device: {}'.format(device))

bb = TrackingBackbone().to(device)


if params.method == 'DKT':
    model = DKT(bb, device)
elif params.method == 'transfer':
    model = FeatureTransfer(bb, device)
else:
    ValueError('Unrecognised method')

optimizer = torch.optim.Adam([{'params': model.model.parameters(), 'lr': 0.001},
                              {'params': model.feature_extractor.parameters(), 'lr': 0.001}])

data = ngsimDataset(os.path.join(configs.data_dir['ngsim'], 'TrainSet.mat'))
train_loader = torch.utils.data.DataLoader(data, batch_size=params.batch_size,
                                           shuffle=True, num_workers=8, collate_fn=data.collate_fn)

for epoch in range(params.stop_epoch):
    for bidx, data in enumerate(train_loader):
        if bidx < 150000:
            model.model.train()
            model.feature_extractor.train()
            #try:
            model.optimize_step(epoch, bidx, optimizer, data)
            if bidx % 100 == 0:
                test_example, pred = model.test_loop(data)
                if test_example is not None:
                    for k in range(5):
                        print(test_example[k]*0.3048)
            # except:
            #     print("Matrix problem: ")
        else:
            break

    model.save_checkpoint(params.checkpoint_dir)
