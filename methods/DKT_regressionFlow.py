## Original packages
## Our packages
import gpytorch
import numpy as np
import torch
import torch.nn as nn

from configs import kernel_type
from data.qmul_loader import get_batch, train_people, test_people
from data_generator import SinusoidalDataGenerator

from utils import normal_logprob
from kernels import NNKernel


def get_transforms(model, use_context):
    if use_context:
        def sample_fn(z, context=None, logpz=None):
            if logpz is not None:
                return model(z, context, logpz, reverse=True)
            else:
                return model(z, context, reverse=True)

        def density_fn(x, context=None, logpx=None):
            if logpx is not None:
                return model(x, context, logpx, reverse=False)
            else:
                return model(x, context, reverse=False)
    else:
        def sample_fn(z, logpz=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z, reverse=True)

        def density_fn(x, logpx=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

    return sample_fn, density_fn


class DKT(nn.Module):
    def __init__(self, backbone, cnf, device, use_conditional):
        super(DKT, self).__init__()
        self.use_conditional = use_conditional
        ## GP parameters
        self.feature_extractor = backbone
        self.cnf = cnf
        self.device = device
        self.get_model_likelihood_mll()  # Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):

        if (train_x is None): train_x = torch.ones(19, 2916).to(self.device)
        if (train_y is None): train_y = torch.ones(19).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
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


    def train_loop(self, epoch, optimizer, params):
        #print("NUM KERNEL PARAMS {}".format(sum([p.numel() for p in self.model.parameters() if p.requires_grad])))
        #print("NUM TRANSFORM PARAMS {}".format(sum([p.numel() for p in self.feature_extractor.parameters() if p.requires_grad])))
        if params.dataset != "sines":
            batch, batch_labels = get_batch(train_people)
        else:
            batch, batch_labels, amp, phase = SinusoidalDataGenerator(params.update_batch_size * 2,
                                                                      params.meta_batch_size,
                                                                      params.output_dim,
                                                                      params.multidimensional_amp,
                                                                      params.multidimensional_phase).generate()
        batch, batch_labels = batch.to(self.device), batch_labels.to(self.device)

        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()
            z = self.feature_extractor(inputs)
            labels = labels.unsqueeze(1)
            if self.use_conditional:
                y, delta_log_py = self.cnf(labels, self.model.kernel.model(z),
                                           torch.zeros(labels.size(0), 1).to(labels))
            else:
                y, delta_log_py = self.cnf(labels, torch.zeros(labels.size(0), 1).to(labels))
            delta_log_py = delta_log_py.view(y.size(0), y.size(1), 1).sum(1)
            y = y.squeeze()
            self.model.set_train_data(inputs=z, targets=y)
            predictions = self.model(z)
            loss = -self.mll(predictions, self.model.train_targets) + torch.mean(delta_log_py)
            loss.backward()
            optimizer.step()
            sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
            if self.use_conditional:
                new_means = sample_fn(predictions.mean.unsqueeze(1), self.model.kernel.model(z))
            else:
                new_means = sample_fn(predictions.mean.unsqueeze(1))
            mse = self.mse(new_means, labels)

            if (epoch % 10 == 0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

    def test_loop(self, n_support, optimizer=None):  # no optimizer needed for GP
        inputs, targets = get_batch(test_people)
        sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
        support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
        query_ind = [i for i in range(19) if i not in support_ind]

        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)

        x_support = inputs[:, support_ind, :, :, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        x_query = inputs[:, query_ind, :, :, :]
        y_query = targets[:, query_ind].to(self.device)

        # choose a random test person
        n = np.random.randint(0, len(test_people) - 1)

        z_support = self.feature_extractor(x_support[n]).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support[n], strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_all[n]).detach()
            pred = self.likelihood(self.model(z_query))
            if self.use_conditional:
                y, delta_log_py = self.cnf(y_all[n].unsqueeze(1), self.model.kernel.model(z_query),
                                           torch.zeros(y_all[n].size(0), 1).to(y_all[n].unsqueeze(1)))
                new_means = sample_fn(pred.mean.unsqueeze(1), self.model.kernel.model(z_query))
            else:
                y, delta_log_py = self.cnf(y_all[n].unsqueeze(1),
                                           torch.zeros(y_all[n].size(0), 1).to(y_all[n].unsqueeze(1)))
                new_means = sample_fn(pred.mean.unsqueeze(1))

            log_py = normal_logprob(y.squeeze(), pred.mean, pred.stddev)

            NLL = -1.0 * torch.mean(log_py - delta_log_py.squeeze())
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        mse = self.mse(new_means, y_all[n])

        return mse, NLL

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict = self.feature_extractor.state_dict()
        cnf_dict = self.cnf.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict,
                    'net': nn_state_dict, 'cnf': cnf_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])
        self.cnf.load_state_dict(ckpt['cnf'])


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        ## RBF kernel
        if (kernel == 'rbf' or kernel == 'RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif (kernel == 'spectral'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916)
        elif(kernel ==  "nn"):
            self.kernel = NNKernel(input_dim=2916, output_dim=16, num_layers=1, hidden_dim=16)
            self.covar_module = self.kernel
        else:
            raise ValueError(
                "[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
