## Original packages
## Our packages
import gpytorch
import numpy as np
import torch
import torch.nn as nn

from configs import kernel_type
from data.qmul_loader import get_batch, train_people, test_people
from data_generator import SinusoidalDataGenerator


from kernels import NNKernel


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class DKT(nn.Module):
    def __init__(self, backbone, device):
        super(DKT, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.device = device
        self.get_model_likelihood_mll()  # Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):

        if (train_x is None): train_x = torch.ones(41, 256).to(self.device)
        if (train_y is None): train_y = torch.ones(41, 2).to(self.device)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel_type)

        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.device)
        self.mse = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass


    def optimize_step(self, epoch, iter, optimizer, data):
        hist, nbrs, mask, _, _, fut, op_mask = data
        if (torch.sum(mask) > 0) & (torch.sum(op_mask) == 50):
            optimizer.zero_grad()
            x = torch.tensor(np.arange(0.0, 3.2, 0.2)).float().to(self.device)
            xf = torch.tensor(np.arange(3.2, 8.2, 0.2)).float().to(self.device)
            x = torch.cat((x, xf), axis=0)
            hist = hist.to(self.device)
            nbrs = nbrs.to(self.device)
            mask = mask.bool().to(self.device)
            fut = fut.to(self.device)
            labels = hist[:, 0, :]
            labelsf = fut[:, 0, :]
            labels = torch.cat((labels, labelsf), axis=0)
            # perm = torch.randperm(x.size(0))#[:16]
            # x = x[perm]
            # labels = labels[perm]
            z = self.feature_extractor(hist, nbrs, mask, x)
            z = z + 0.01 * torch.randn(z.size()).cuda()
            labels = labels + 0.01 * torch.randn(labels.size()).cuda()
            self.model.set_train_data(inputs=z, targets=labels, strict=False)
            predictions = self.model(z)
            try:
                loss = -self.mll(predictions, self.model.train_targets)
            except:
                loss = self.mse(predictions.mean, labels)
            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels)
            if (iter % 100 == 0):
                print('[%d][%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, iter, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

    def test_loop(self, data):  # no optimizer needed for GP
        hist, nbrs, mask, _, _, fut, op_mask = data
        if (torch.sum(mask) > 0) & (torch.sum(op_mask) == 50):
            x = torch.tensor(np.arange(0.0, 3.2, 0.2)).float().to(self.device)
            xf = torch.tensor(np.arange(3.2, 8.2, 0.2)).float().to(self.device)
            hist = hist.to(self.device)
            nbrs = nbrs.to(self.device)
            mask = mask.bool().to(self.device)
            fut = fut.to(self.device)
            z = self.feature_extractor(hist, nbrs, mask, x)
            labels = hist[:, 0, :]
            labelsf = fut[:, 0, :]
            self.model.set_train_data(inputs=z, targets=labels, strict=False)
            self.model.eval()
            self.feature_extractor.eval()
            self.likelihood.eval()

            with torch.no_grad():
                z_query = self.feature_extractor(hist, nbrs, mask, xf).detach()
                pred = self.likelihood(self.model(z_query))
                lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean
            TIMES = [4, 9, 14, 19, 24]
            MSES = []
            for k in TIMES:
                mse = torch.sqrt(torch.pow(pred.mean[k,0] - labelsf[k,0], 2) + torch.pow(pred.mean[k,1] - labelsf[k,1], 2))
                MSES.append(mse.item())
            return MSES, pred.mean
        else:
            return None, None

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net': nn_state_dict}, checkpoint)


    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        ## RBF kernel
        if (kernel == 'rbf' or kernel == 'RBF'):
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif (kernel == 'spectral'):
            kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916)
        elif(kernel ==  "nn"):
            kernel = NNKernel(input_dim=256, output_dim=16, num_layers=1, hidden_dim=16)
        else:
            raise ValueError(
                "[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")
        self.covar_module = gpytorch.kernels.MultitaskKernel(kernel, num_tasks=2, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class HyperFlowNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = highwayNet()
        output = []
        self.n_out = 128
        # self.n_out = 46080
        dims = tuple(map(int, args.dims.split("-")))
        for k in range(len(dims)):
            if k == 0:
                output.append(nn.Linear(self.n_out, args.input_dim * dims[k], bias=True))
            else:
                output.append(nn.Linear(self.n_out, dims[k - 1] * dims[k], bias=True))
            #bias
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            #scaling
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            #shift
            output.append(nn.Linear(self.n_out, dims[k], bias=True))

        output.append(nn.Linear(self.n_out, dims[-1] * args.input_dim, bias=True))
        # bias
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        # scaling
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))
        # shift
        output.append(nn.Linear(self.n_out, args.input_dim, bias=True))

        self.output = ListModule(*output)

    def forward(self, hist, nbrs, masks):
        output = self.encoder(hist, nbrs, masks)
        # output = output.view(output.size(0), -1)
        multi_outputs = []
        for j, target_network_layer in enumerate(self.output):
            multi_outputs.append(target_network_layer(output))
        multi_outputs = torch.cat(multi_outputs, dim=1)
        return multi_outputs


class highwayNet(nn.Module):

    ## Initialization
    def __init__(self):
        super(highwayNet, self).__init__()

        ## Sizes of network layers
        self.encoder_size = 64
        self.decoder_size = 128
        self.grid_size = (13, 3)
        self.soc_conv_depth = 64
        self.conv_3x1_depth = 16
        self.dyn_embedding_size = 32
        self.input_embedding_size = 32
        self.soc_embedding_size = (((self.grid_size[0]-4)+1)//2)*self.conv_3x1_depth

        ## Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3)
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1))
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0))

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()

        # self.fc1 = nn.Linear(in_features=112, out_features=32, bias=True)
        self.fc1 = nn.Linear(in_features=112, out_features=128, bias=True)
        #self.fc2 = nn.Linear(in_features=512, out_features=1024, bias=True)

    ## Forward Pass
    def forward(self,hist,nbrs,masks):

        ## Forward pass hist:
        _,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

        ## Forward pass nbrs
        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ## Masked scatter
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0,3,2,1)

        ## Apply convolutional social pooling:
        soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)

        ## Concatenate encodings:
        enc = nn.functional.relu(torch.cat((soc_enc,hist_enc),1))

        out_fc1 = nn.functional.relu(self.fc1(enc))
        #out_fc2 = nn.functional.relu(self.fc2(out_fc1))

        return out_fc1


class TrackingBackbone(nn.Module):

    def __init__(self, input_size=1):
        super(TrackingBackbone, self).__init__()

        self.encoder = highwayNet()
        self.fc1 = nn.Linear(input_size, 128, bias=True)
        self.fc2 = nn.Linear(256, 512, bias=True)
        self.fc3 = nn.Linear(512, 1024, bias=True)
        self.fc4 = nn.Linear(1024, 256, bias=True)

    def forward(self, hist, nbrs, masks, x):
        z = self.encoder(hist, nbrs, masks)
        z = z.expand((x.shape[0], z.shape[1]))
        feature_embedding = nn.functional.relu(self.fc1(torch.unsqueeze(x, 1)))
        z1 = nn.functional.relu(self.fc2(torch.cat((z, feature_embedding), axis=1)))
        z1 = nn.functional.relu(self.fc3(z1))
        z2 = nn.functional.relu(self.fc4(z1))
        return z2

