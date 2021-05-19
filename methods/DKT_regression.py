## Original packages
## Our packages
import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.constraints import GreaterThan
from gpytorch.priors import UniformPrior
from torch.utils.data import DataLoader

from data.data_generator import SinusoidalDataGenerator, Nasdaq100padding, PowerConsumption
from data.qmul_loader import get_batch, train_people, test_people
from models.kernels import NNKernel, MultiNNKernel
from training.utils import prepare_for_plots, plot_histograms


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
    def __init__(self, backbone, device, num_tasks=1, config=None, dataset='QMUL', cnf=None, use_conditional=False,
                 add_noise=False, context_type='nn', multi_type=2):
        super(DKT, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.device = device
        self.num_tasks = num_tasks
        self.config = config
        self.dataset = dataset
        self.cnf = cnf
        self.use_conditional = use_conditional
        self.multi_type = multi_type
        self.context_type = context_type
        if self.cnf is not None:
            self.is_flow = True
        else:
            self.is_flow = False
        self.add_noise = add_noise
        self.get_model_likelihood_mll()  # Init model, likelihood, and mll

    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if self.dataset == 'QMUL':
            if train_x is None: train_x = torch.ones(19, 2916).to(self.device)
            if train_y is None: train_y = torch.ones(19).to(self.device)
        elif self.dataset == "sines":
            if self.num_tasks == 1:
                if train_x is None: train_x = torch.ones(10, self.feature_extractor.output_dim).to(self.device)
                if train_y is None: train_y = torch.ones(10).to(self.device)
            else:
                if train_x is None: train_x = torch.ones(10, self.feature_extractor.output_dim).to(self.device)
                if train_y is None: train_y = torch.ones(10, self.num_tasks).to(self.device)
        elif self.dataset == "nasdaq" or self.dataset == "eeg" or self.dataset == "power":
            if self.num_tasks == 1:
                if train_x is None: train_x = torch.ones(10, self.feature_extractor.output_dim).to(self.device)
                if train_y is None: train_y = torch.ones(10).to(self.device)
            else:
                if train_x is None: train_x = torch.ones(10, self.feature_extractor.output_dim).to(self.device)
                if train_y is None: train_y = torch.ones(10, self.num_tasks).to(self.device)

        if self.num_tasks == 1:
            noise_prior = UniformPrior(0, 1)
            MIN_INFERRED_NOISE_LEVEL = 1e-8
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    MIN_INFERRED_NOISE_LEVEL
                ),
            )
            # noise fixed to 0s'
            # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.zeros(10),
            #                                                                learn_additional_noise=False)
            model = ExactGPLayer(dataset=self.dataset, config=self.config, train_x=train_x,
                                 train_y=train_y, likelihood=likelihood, kernel=self.config.kernel_type)
        else:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks)
            model = MultitaskExactGPLayer(dataset=self.dataset, config=self.config, train_x=train_x, train_y=train_y,
                                          likelihood=likelihood,
                                          kernel=self.config.kernel_type, num_tasks=self.num_tasks,
                                          multi_type=self.multi_type)
        self.model = model.to(self.device)
        self.likelihood = likelihood.to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.device)
        self.mse = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def train_loop(self, epoch, optimizer, params, results_logger):
        # print("NUM KERNEL PARAMS {}".format(sum([p.numel() for p in self.model.parameters() if p.requires_grad])))
        # print("NUM TRANSFORM PARAMS {}".format(sum([p.numel() for p in self.feature_extractor.parameters() if p.requires_grad])))

        # fighting with numerical problems with cholesky decomposition
        with gpytorch.settings.cholesky_jitter(1e-4, 1e-5):
            self.model.train()
            self.feature_extractor.train()
            self.likelihood.train()
            if self.is_flow:
                self.cnf.train()

            if self.dataset == "sines":
                batch, batch_labels, amp, phase = SinusoidalDataGenerator(params.update_batch_size * 2,
                                                                          params.meta_batch_size,
                                                                          params.num_tasks,
                                                                          params.multidimensional_amp,
                                                                          params.multidimensional_phase,
                                                                          params.noise).generate()

                if self.num_tasks == 1:
                    batch = torch.from_numpy(batch)
                    batch_labels = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)
                else:
                    batch = torch.from_numpy(batch)
                    batch_labels = torch.from_numpy(batch_labels)
                print(batch_labels.shape)
            elif self.dataset == "nasdaq" or self.dataset == "eeg":
                nasdaq100padding = Nasdaq100padding(directory=self.config.data_dir['nasdaq'], normalize=True,
                                                    partition="train", window=params.update_batch_size * 2,
                                                    time_to_predict=params.meta_batch_size * 2)
                data_loader = torch.utils.data.DataLoader(nasdaq100padding, batch_size=params.update_batch_size * 2,
                                                          shuffle=True)
                batch, batch_labels = next(iter(data_loader))
                batch = batch.reshape(params.update_batch_size * 2, params.meta_batch_size * 2, 1)
                batch_labels = batch_labels[:, :, -1].float()
            elif self.dataset == "power":
                power = PowerConsumption(directory=self.config.data_dir['power'], partition="train")
                dataloader = DataLoader(power, batch_size=params.update_batch_size * 2, shuffle=True, num_workers=0)
            else:
                batch, batch_labels = get_batch(train_people)

            # batch, batch_labels = batch.to(self.device), batch_labels.to(self.device)
            # print(batch.shape, batch_labels.shape)
            # for inputs, labels in zip(batch, batch_labels):
            for _, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                labels = labels[:, -1].float()
                z = self.feature_extractor(inputs.reshape(inputs.shape[0], 1).float())
                if self.add_noise:
                    labels = labels + torch.normal(0, 0.1, size=labels.shape).to(labels)
                if self.is_flow:
                    delta_log_py, labels, y = self.apply_flow(labels, z)
                else:
                    y = labels
                self.model.set_train_data(inputs=z, targets=y)
                predictions = self.model(z)
                loss = -self.mll(predictions, self.model.train_targets)
                if self.is_flow:
                    loss = loss + torch.mean(delta_log_py)
                loss.backward()
                optimizer.step()

                mse, _ = self.compute_mse(labels, predictions, z)

                if epoch % 10 == 0:
                    print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                        epoch, loss.item(), mse.item(),
                        self.model.likelihood.noise[0].item()
                    ))
                    results_logger.log("epoch", epoch)
                    results_logger.log("loss", loss.item())
                    results_logger.log("MSE", mse.item())
                    results_logger.log("noise", self.model.likelihood.noise[0].item())
            return loss.item()

    def compute_mse(self, labels, predictions, z):
        if self.is_flow:
            sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
            if self.num_tasks == 1:
                means = predictions.mean.unsqueeze(1)
            else:
                means = predictions.mean
            if self.use_conditional:
                new_means = sample_fn(means, self.get_context(z))
            else:
                new_means = sample_fn(means)
            mse = self.mse(new_means.squeeze(), labels)
        else:
            mse = self.mse(predictions.mean, labels)
            new_means = None
        return mse, new_means

    def get_context(self, z):
        if self.context_type == 'nn':
            if self.num_tasks == 1:
                context = self.model.kernel.model(z)
            else:
                if self.multi_type == 3:
                    contexts = []
                    for k in range(len(self.model.kernels)):
                        contexts.append(self.model.kernels[k].model(z))
                    context = sum(contexts)
                else:
                    context = self.model.kernels.model(z)
        elif self.context_type == 'backbone':
            context = z
        else:
            raise ValueError("unknown context type")
        return context

    def apply_flow(self, labels, z):
        if self.num_tasks == 1:
            labels = labels.unsqueeze(1)
        if self.use_conditional:
            y, delta_log_py = self.cnf(labels, self.get_context(z),
                                       torch.zeros(labels.size(0), 1).to(labels))
        else:
            y, delta_log_py = self.cnf(labels, torch.zeros(labels.size(0), 1).to(labels))
        y = y.squeeze()
        return delta_log_py, labels, y

    def test_loop(self, n_support, params=None, save_dir=None):
        if self.dataset == "sines":
            x_all, x_support, y_all, y_support = self.get_support_query_sines(n_support, params)
        elif self.dataset == "nasdaq" or self.dataset == "eeg":
            x_all, x_support, y_all, y_support = self.get_support_query_nasdaq(n_support, params)
        elif self.dataset == "power":
            x_all, x_support, y_all, y_support = self.get_support_query_power(n_support, params)
        elif params is None:
            x_all, x_support, y_all, y_support = self.get_support_query_qmul(n_support)
        else:
            raise ValueError("unknown dataset")

        sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
        # choose a random test person
        n = np.random.randint(0, x_support.shape[0])
        z_support = self.feature_extractor(x_support[n]).detach()
        labels = y_support[n]

        if self.is_flow:
            with torch.no_grad():
                _, labels, y_support = self.apply_flow(labels, z_support)
        else:
            y_support = labels

        self.model.set_train_data(inputs=z_support, targets=y_support, strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()
        if self.is_flow:
            self.cnf.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_all[n]).detach()
            predictions_query = self.model(z_query)
            pred = self.likelihood(predictions_query)
            context = None
            new_means = None
            if self.is_flow:
                if self.num_tasks == 1:
                    mean_base = pred.mean.unsqueeze(1)
                else:
                    mean_base = pred.mean
                if self.use_conditional:
                    context = self.get_context(z_query)
                    new_means = sample_fn(mean_base, context)
                else:
                    new_means = sample_fn(mean_base)
                delta_log_py, _, y = self.apply_flow(y_all[n], z_query)
                log_py = -self.mll(predictions_query, y.squeeze())
                NLL = log_py + torch.mean(delta_log_py.squeeze())
                # log_py = normal_logprob(y.squeeze(), pred.mean, pred.stddev)
                # NLL = -1.0 * torch.mean(log_py - delta_log_py.squeeze())
            else:
                NLL = -self.mll(predictions_query, torch.tensor([y_all[n]]))
                # log_py = normal_logprob(y_all[n], pred.mean, pred.stddev)
                # NLL = -1.0 * torch.mean(log_py)
            if save_dir is not None and self.num_tasks == 1:
                samples, true_y, gauss_y, flow_samples, flow_y = prepare_for_plots(pred, torch.tensor([y_all[n]]),
                                                                                   sample_fn, context, new_means)
                plot_histograms(save_dir, samples, true_y, gauss_y, n, flow_samples, flow_y)
            mse, new_means = self.compute_mse(torch.tensor([y_all[n]]), pred, z_query)
            lower, upper = pred.confidence_region()  # 2 standard deviations above and below the mean

        if self.is_flow:
            return mse, NLL, new_means, lower, upper, x_all[n], y_all[n]
        else:
            return mse, NLL, pred.mean, lower, upper, x_all[n], y_all[n]

    def get_support_query_qmul(self, n_support):
        inputs, targets = get_batch(test_people)
        support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
        query_ind = [i for i in range(19) if i not in support_ind]
        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)
        x_support = inputs[:, support_ind, :, :, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        x_query = inputs[:, query_ind, :, :, :]
        y_query = targets[:, query_ind].to(self.device)
        return x_all, x_support, y_all, y_support

    def get_support_query_sines(self, n_support, params):
        batch, batch_labels, amp, phase = SinusoidalDataGenerator(params.update_batch_size * 2,
                                                                  params.meta_batch_size,
                                                                  params.num_tasks,
                                                                  params.multidimensional_amp,
                                                                  params.multidimensional_phase,
                                                                  params.noise).generate()
        if self.num_tasks == 1:
            inputs = torch.from_numpy(batch)
            targets = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)
        else:
            inputs = torch.from_numpy(batch)
            targets = torch.from_numpy(batch_labels)

        support_ind = list(np.random.choice(list(range(10)), replace=False, size=n_support))
        query_ind = [i for i in range(10) if i not in support_ind]
        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)
        x_support = inputs[:, support_ind, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        return x_all, x_support, y_all, y_support

    def get_support_query_power(self, n_support, params):
        power_consumption = PowerConsumption(directory=self.config.data_dir['power'], partition="test")
        data_loader = torch.utils.data.DataLoader(power_consumption, batch_size=params.update_batch_size * 2,
                                                  shuffle=True)
        batch, batch_labels = next(iter(data_loader))
        inputs = batch.reshape(params.update_batch_size * 2, 1)
        targets = batch_labels[:, -1].float()

        # if self.num_tasks == 1:
        #     inputs = torch.from_numpy(batch)
        #     targets = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)
        # else:
        #     inputs = torch.from_numpy(batch)
        #     targets = torch.from_numpy(batch_labels)

        support_ind = list(np.random.choice(list(range(10)), replace=True, size=n_support))
        query_ind = [i for i in range(10) if i not in support_ind]
        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)
        x_support = inputs[support_ind, :].to(self.device)
        y_support = targets[support_ind].to(self.device)
        return x_all.float(), x_support.float(), y_all.float(), y_support.float()

    def get_support_query_nasdaq(self, n_support, params):
        nasdaq100padding = Nasdaq100padding(directory=self.config.data_dir['nasdaq'], normalize=True, partition="train",
                                            window=params.update_batch_size * 2,
                                            time_to_predict=params.meta_batch_size * 2)
        data_loader = torch.utils.data.DataLoader(nasdaq100padding, batch_size=params.update_batch_size * 2,
                                                  shuffle=True)
        batch, batch_labels = next(iter(data_loader))
        inputs = batch.reshape(params.update_batch_size * 2, params.meta_batch_size * 2, 1)
        targets = batch_labels[:, :, -1].float()

        # if self.num_tasks == 1:
        #     inputs = torch.from_numpy(batch)
        #     targets = torch.from_numpy(batch_labels).view(batch_labels.shape[0], -1)
        # else:
        #     inputs = torch.from_numpy(batch)
        #     targets = torch.from_numpy(batch_labels)

        support_ind = list(np.random.choice(list(range(10)), replace=True, size=n_support))
        query_ind = [i for i in range(10) if i not in support_ind]
        x_all = inputs.to(self.device)
        y_all = targets.to(self.device)
        x_support = inputs[:, support_ind, :].to(self.device)
        y_support = targets[:, support_ind].to(self.device)
        return x_all, x_support, y_all, y_support

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict = self.feature_extractor.state_dict()

        state_dicts = {'gp': gp_state_dict, 'likelihood': likelihood_state_dict,
                       'net': nn_state_dict}
        if self.is_flow:
            cnf_dict = self.cnf.state_dict()
            state_dicts['cnf'] = cnf_dict
        torch.save(state_dicts, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])
        if self.is_flow:
            self.cnf.load_state_dict(ckpt['cnf'])


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, dataset, config, train_x, train_y, likelihood, kernel='linear'):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.dataset = dataset
        ## RBF kernel
        if kernel == 'rbf' or kernel == 'RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Spectral kernel
        elif kernel == 'spectral':
            if self.dataset == "sines":
                ard_num_dims = 1
            elif self.dataset == "nasdaq" or self.dataset == "eeg":
                ard_num_dims = 1
            else:
                ard_num_dims = 2916
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=ard_num_dims)
        elif kernel == "nn":
            # self.kernel = NNKernelNoInner(input_dim=config.nn_config["input_dim"],
            #                        num_layers=config.nn_config["num_layers"],
            #                        hidden_dim=config.nn_config["hidden_dim"])

            self.kernel = NNKernel(input_dim=config.nn_config["input_dim"],
                                   output_dim=config.nn_config["output_dim"],
                                   num_layers=config.nn_config["num_layers"],
                                   hidden_dim=config.nn_config["hidden_dim"])

            self.covar_module = self.kernel
        else:
            raise ValueError(
                "[ERROR] the kernel '" + str(kernel) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, dataset, config, train_x, train_y, likelihood, kernel='nn', num_tasks=2, multi_type=2):
        super(MultitaskExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.dataset = dataset
        if kernel == "nn":
            if multi_type == 2:
                self.kernels = NNKernel(input_dim=config.nn_config["input_dim"],
                                        output_dim=config.nn_config["output_dim"],
                                        num_layers=config.nn_config["num_layers"],
                                        hidden_dim=config.nn_config["hidden_dim"])

                self.covar_module = gpytorch.kernels.MultitaskKernel(self.kernels, num_tasks)
            elif multi_type == 3:
                self.kernels = []
                for i in range(num_tasks):
                    self.kernels.append(NNKernel(input_dim=config.nn_config["input_dim"],
                                                 output_dim=config.nn_config["output_dim"],
                                                 num_layers=config.nn_config["num_layers"],
                                                 hidden_dim=config.nn_config["hidden_dim"]))
                self.covar_module = MultiNNKernel(num_tasks, self.kernels)
            else:
                raise ValueError("Unsupported multi kernel type {}".format(multi_type))
        elif kernel == "rbf":

            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
            )
        else:
            raise ValueError(
                "[ERROR] the kernel '" + str(kernel) + "' is not supported for multi-regression, use 'nn'.")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
