## Original packages
## Our packages
import gpytorch
import numpy as np
import torch
import torch.nn as nn

from configs import kernel_type
from data.qmul_loader import get_batch, train_people, test_people
from data.data_generator import SinusoidalDataGenerator, PolynomialDataGenerator

from kernels import NNKernel


class DKT(nn.Module):
    def __init__(self, backbone, params, device):
        super(DKT, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.device = device
        self.dataset = params.dataset
        self.context = params.context
        self.get_model_likelihood_mll()  # Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):

        # if (train_x is None): train_x = torch.ones(19, 2916).to(self.device)
        # if (train_y is None): train_y = torch.ones(19).to(self.device)
        if self.dataset == "polynomials" and self.context is True:
            # if (train_x is None): train_x = torch.ones(10, 1).to(self.device)
            # if (train_x is None): train_x = torch.ones(10, 6).to(self.device)
            # different latent space's dimensionality
            if (train_x is None): train_x = torch.ones(10, 15).to(self.device)
        else:
            # if (train_x is None): train_x = torch.ones(10, 1).to(self.device)
            if (train_x is None): train_x = torch.ones(10, 15).to(self.device)

        if (train_y is None): train_y = torch.ones(10).to(self.device)

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
        # print("NUM KERNEL PARAMS {}".format(sum([p.numel() for p in self.model.parameters() if p.requires_grad])))
        # print("NUM TRANSFORM PARAMS {}".format(sum([p.numel() for p in self.feature_extractor.parameters() if p.requires_grad])))
        if params.dataset == "polynomials":
            batch, batch_labels, degrees = PolynomialDataGenerator(params.update_batch_size * 2,
                                                                   params.meta_batch_size,
                                                                   params.output_dim,
                                                                   params.context).generate()

            batch = torch.from_numpy(batch)
            batch_labels = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)
        elif params.dataset == "sines":
            batch, batch_labels, amp, phase = SinusoidalDataGenerator(params.update_batch_size * 2,
                                                                      params.meta_batch_size,
                                                                      params.output_dim,
                                                                      params.multidimensional_amp,
                                                                      params.multidimensional_phase).generate()
            batch = torch.from_numpy(batch)
            batch_labels = torch.from_numpy(batch_labels)
        else:

            batch, batch_labels = get_batch(train_people)

        batch, batch_labels = batch.to(self.device), batch_labels.to(self.device)

        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()
            context_to_add = inputs[:, 1:]
            z = self.feature_extractor(inputs)

            z_with_context = torch.cat((z, context_to_add), 1)

            #TODO - maybe self.model.set_train_data(inputs=z, targets=labels, strict=False)
            # self.model.set_train_data(inputs=z, targets=labels)
            # predictions = self.model(z)
            self.model.set_train_data(inputs=z_with_context, targets=labels)
            predictions = self.model(z_with_context)
            loss = -self.mll(predictions, self.model.train_targets)

            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels)

            if (epoch % 10 == 0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))

    def test_loop(self, n_support, optimizer=None, params=None):
        if params.dataset == "sines":
            return self.test_loop_sines(n_support, params, optimizer)
        elif params.dataset == "polynomials":
            return self.test_loop_polynomials(n_support, params, optimizer)
        elif params is None or params.dataset not in ["sines", "polynomials"]:
            return self.test_loop_qmul(n_support, optimizer)
        else:
            raise ValueError("unknown dataset")

    def test_loop_qmul(self, n_support, optimizer=None):  # no optimizer needed for GP
        inputs, targets = get_batch(test_people)

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
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_all[n])

        return mse

    def test_loop_polynomials(self, n_support, params, optimizer=None):  # no optimizer needed for GP
        batch, batch_labels, degrees = PolynomialDataGenerator(params.update_batch_size * 2,
                                                                  params.meta_batch_size,
                                                                  params.output_dim,
                                                                  params.context).generate()
        inputs = torch.from_numpy(batch)
        targets = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)

        support_ind = list(np.random.choice(list(range(10)), replace=False, size=n_support))
        query_ind = [i for i in range(10) if i not in support_ind]

        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)

        x_support = inputs[:, support_ind, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        x_query = inputs[:, query_ind, :]
        y_query = targets[:, query_ind].to(self.device)

        # choose a random test person
        n = np.random.randint(0, x_support.shape[0])

        z_support = self.feature_extractor(x_support[n]).detach()
        context_to_add = x_support[n][:, 1:]
        # print(x_support[n][:, 1:])
        z_support_with_context = torch.cat((z_support, context_to_add), 1)
        # self.model.set_train_data(inputs=z_support, targets=y_support[n], strict=False)
        self.model.set_train_data(inputs=z_support_with_context, targets=y_support[n], strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_all[n]).detach()
            context_to_add_all = x_all[n][:, 1:]
            z_query_with_context = torch.cat((z_query, context_to_add_all), 1)
            # pred = self.likelihood(self.model(z_query))
            pred = self.likelihood(self.model(z_query_with_context))
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_all[n])

        return mse

    def test_loop_sines(self, n_support, params, optimizer=None):  # no optimizer needed for GP
        batch, batch_labels, amp, phase = SinusoidalDataGenerator(params.update_batch_size * 2,
                                                                  params.meta_batch_size,
                                                                  params.output_dim,
                                                                  params.multidimensional_amp,
                                                                  params.multidimensional_phase).generate()
        inputs = torch.from_numpy(batch)
        targets = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)

        support_ind = list(np.random.choice(list(range(10)), replace=False, size=n_support))
        query_ind = [i for i in range(10) if i not in support_ind]

        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)

        x_support = inputs[:, support_ind, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        x_query = inputs[:, query_ind, :]
        y_query = targets[:, query_ind].to(self.device)

        # choose a random test person
        n = np.random.randint(0, x_support.shape[0])

        z_support = self.feature_extractor(x_support[n]).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support[n], strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_all[n]).detach()
            pred = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_all[n])

        return mse

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
        self.mean_module = gpytorch.means.ConstantMean()

        ## RBF kernel
        if (kernel == 'rbf' or kernel == 'RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif (kernel == 'spectral'):
            # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916)
            # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=1)
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=100, ard_num_dims=10)
            # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=100, ard_num_dims=6)
        elif (kernel == "nn"):
            # kernel = NNKernel(input_dim=2916, output_dim=16, num_layers=1, hidden_dim=16)
            # kernel = NNKernel(input_dim=1, output_dim=1, num_layers=2, hidden_dim=16)
            kernel = NNKernel(input_dim=15, output_dim=1, num_layers=2, hidden_dim=16)
            # kernel = NNKernel(input_dim=15, output_dim=1, num_layers=4, hidden_dim=40)
            # kernel = NNKernel(input_dim=6, output_dim=1, num_layers=4, hidden_dim=40)
            self.covar_module = kernel
        else:
            raise ValueError(
                "[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
