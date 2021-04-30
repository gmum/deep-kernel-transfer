## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from utils import normal_logprob

## Our packages
import gpytorch
from time import gmtime, strftime
import random


from kernels import NNKernel
#Check if tensorboardx is installed
try:
    from tensorboardX import SummaryWriter
    IS_TBX_INSTALLED = True
except ImportError:
    IS_TBX_INSTALLED = False
    print('[WARNING] install tensorboardX to record simulation logs.')


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


## Training CMD
#ATTENTION: to test each method use exaclty the same command but replace 'train.py' with 'test.py'
# Omniglot->EMNIST without data augmentation
#python3 train.py --dataset="cross_char" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1
#python3 train.py --dataset="cross_char" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=5
# CUB + data augmentation
#python3 train.py --dataset="CUB" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=1 --train_aug
#python3 train.py --dataset="CUB" --method="DKT" --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug

class DKT(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, cnf, use_conditional, config=None):
        super(DKT, self).__init__(model_func, n_way, n_support)
        self.config = config
        ## GP parameters
        self.leghtscale_list = None
        self.noise_list = None
        self.outputscale_list = None
        self.iteration = 0
        self.writer=None
        self.feature_extractor = self.feature
        self.cnf = cnf
        self.use_conditional = use_conditional
        self.get_model_likelihood_mll() #Init model, likelihood, and mll
        if(self.config.kernel_type=="cossim"):
            self.normalize=True
        elif(self.config.kernel_type=="bncossim"):
            self.normalize=True
            latent_size = np.prod(self.feature_extractor.final_feat_dim)
            self.feature_extractor.trunk.add_module("bn_out", nn.BatchNorm1d(latent_size))
        else:
            self.normalize=False

    def init_summary(self):
        if(IS_TBX_INSTALLED):
            time_string = strftime("%d%m%Y_%H%M%S", gmtime())
            writer_path = "./log/" + time_string
            self.writer = SummaryWriter(log_dir=writer_path)

    def get_model_likelihood_mll(self, train_x_list=None, train_y_list=None):
        if(train_x_list is None): train_x_list=[torch.ones(100, 64).cuda()]*self.n_way
        if(train_y_list is None): train_y_list=[torch.ones(100).cuda()]*self.n_way
        model_list = list()
        likelihood_list = list()
        for train_x, train_y in zip(train_x_list, train_y_list):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPLayer(config = self.config, train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=self.config.kernel_type)
            model_list.append(model)
            likelihood_list.append(model.likelihood)
        self.model = gpytorch.models.IndependentModelList(*model_list).cuda()
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list).cuda()
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model).cuda()
        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def _reset_likelihood(self, debug=False):
        for param in self.likelihood.parameters():
           param.data.normal_(0.0, 0.01)

    def _print_weights(self):
        for k, v in self.feature_extractor.state_dict().items():
            print("Layer {}".format(k))
            print(v)

    def _reset_variational(self):
        mean_init = torch.zeros(128) #num_inducing_points
        covar_init = torch.eye(128, 128) #num_inducing_points
        mean_init = mean_init.repeat(64, 1) #batch_shape
        covar_init = covar_init.repeat(64, 1, 1) #batch_shape
        for idx, param in enumerate(self.gp_layer.variational_parameters()):
            if(idx==0): param.data.copy_(mean_init) #"variational_mean"
            elif(idx==1): param.data.copy_(covar_init) #"chol_variational_covar"
            else: raise ValueError('[ERROR] DKT the variational_parameters at index>1 should not exist!')

    def _reset_parameters(self):
        if(self.leghtscale_list is None):
            self.leghtscale_list = list()
            self.noise_list = list()
            self.outputscale_list = list()
            for idx, single_model in enumerate(self.model.models):
                self.leghtscale_list.append(single_model.covar_module.base_kernel.lengthscale.clone().detach())
                self.noise_list.append(single_model.likelihood.noise.clone().detach())
                self.outputscale_list.append(single_model.covar_module.outputscale.clone().detach())
        else:
            for idx, single_model in enumerate(self.model.models):
                single_model.covar_module.base_kernel.lengthscale=self.leghtscale_list[idx].clone().detach()#.requires_grad_(True)
                single_model.likelihood.noise=self.noise_list[idx].clone().detach()
                single_model.covar_module.outputscale=self.outputscale_list[idx].clone().detach()

    def train_loop(self, epoch, train_loader, optimizer, print_freq=10):
        optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': 1e-4},
                                      {'params': self.feature_extractor.parameters(), 'lr': 1e-3},
                                      {'params': self.cnf.parameters(), 'lr': 1e-3}])

        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way: self.n_way  = x.size(0)
            x_all = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]).cuda()
            y_all = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query+self.n_support)).cuda())
            x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
            y_support = np.repeat(range(self.n_way), self.n_support)
            x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
            y_query = np.repeat(range(self.n_way), self.n_query)
            x_train = x_all
            y_train = y_all

            target_list = list()
            flow_target_list = list()
            flow_delta_log_py_list = list()
            samples_per_model = int(len(y_train) / self.n_way) #25 / 5 = 5

            self.model.train()
            self.likelihood.train()
            self.feature_extractor.train()
            z_train = self.feature_extractor.forward(x_train)
            if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)

            for way in range(self.n_way):
                # print('WAY: ', way)
                target = torch.ones(len(y_train), dtype=torch.float32) * -1.0

                start_index = way * samples_per_model
                stop_index = start_index+samples_per_model
                target[start_index:stop_index] = 1.0

                target_list.append(target.unsqueeze(1).cuda())

            y = torch.cat(target_list, axis=1).unsqueeze(1)
            y = y + torch.randn(y.size()).to(y)
            y, delta_log_py = self.cnf(y, z_train.detach(),
                                       torch.zeros(y.size(0), y.size(1), 1).to(y))
            delta_log_py = delta_log_py.view(y.size(0), y.size(1), 1).sum(1)
            train_list = [z_train]*self.n_way
            lenghtscale = 0.0
            noise = 0.0
            outputscale = 0.0
            for idx, single_model in enumerate(self.model.models):
                single_model.set_train_data(inputs=z_train, targets=y[:,0,idx], strict=False)

                if single_model.covar_module.base_kernel.lengthscale is not None:
                    lenghtscale += single_model.covar_module.base_kernel.lengthscale.mean().cpu().detach().numpy().squeeze()
                noise += single_model.likelihood.noise.cpu().detach().numpy().squeeze()

                if single_model.covar_module.outputscale is not None:
                    outputscale+=single_model.covar_module.outputscale.cpu().detach().numpy().squeeze()
            if single_model.covar_module.base_kernel.lengthscale is not None:
                lenghtscale /= float(len(self.model.models))

            noise /= float(len(self.model.models))
            if single_model.covar_module.outputscale is not None:
                outputscale /= float(len(self.model.models))

            ## Optimize
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            #TODO - consider if we should use mean or sum function
            # loss = -self.mll(output, self.model.train_targets) + torch.mean(torch.tensor(flow_delta_list))
            loss = -self.mll(output, self.model.train_targets) + torch.sum(delta_log_py)
            # loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()

            self.iteration = i+(epoch*len(train_loader))
            if(self.writer is not None): self.writer.add_scalar('loss', loss, self.iteration)

            #Eval on the query (validation set)
            sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
            with torch.no_grad():
                self.cnf.eval()
                self.model.eval()
                self.likelihood.eval()
                self.feature_extractor.eval()
                z_support = self.feature_extractor.forward(x_support).detach()
                if (self.normalize): z_support = F.normalize(z_support, p=2, dim=1)
                z_support_list = [z_support] * len(y_support)
                predictions = self.likelihood(*self.model(*z_support_list))  # return 20 MultiGaussian Distributions
                gaussian_means = []
                for gaussian in predictions:
                    gaussian_means.append(gaussian.mean.unsqueeze(1))
                gaussian_means = torch.cat(gaussian_means, axis=1).unsqueeze(1)
                preds = sample_fn(gaussian_means, z_support).squeeze()
                predictions_list = torch.sigmoid(preds).cpu().detach().numpy()
                y_pred = predictions_list.argmax(axis=1)  # [model, classes]
                accuracy_support = (np.sum(y_pred == y_support) / float(len(y_support))) * 100.0
                if (self.writer is not None): self.writer.add_scalar('GP_support_accuracy', accuracy_support,
                                                                     self.iteration)
                z_query = self.feature_extractor.forward(x_query).detach()
                if (self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
                z_query_list = [z_query] * len(y_query)
                predictions = self.likelihood(*self.model(*z_query_list))  # return 20 MultiGaussian Distributions
                gaussian_means = []
                for gaussian in predictions:
                    gaussian_means.append(gaussian.mean.unsqueeze(1))
                gaussian_means = torch.cat(gaussian_means, axis=1).unsqueeze(1)
                preds = sample_fn(gaussian_means, z_query).squeeze()
                predictions_list = torch.sigmoid(preds).cpu().detach().numpy()
                y_pred = predictions_list.argmax(axis=1)  # [model, classes]
                accuracy_query = (np.sum(y_pred == y_query) / float(len(y_query))) * 100.0
                if (self.writer is not None): self.writer.add_scalar('GP_query_accuracy', accuracy_query,
                                                                     self.iteration)

            if i % print_freq == 0:
                if (self.writer is not None): self.writer.add_histogram('z_support', z_support, self.iteration)
                print(
                    'Epoch [{:d}] [{:d}/{:d}] | Outscale {:f} | Lenghtscale {:f} | Noise {:f} | Loss {:f} | Supp. {:f} | Query {:f}'.format(
                        epoch, i, len(train_loader), outputscale, lenghtscale, noise, loss.item(), accuracy_support,
                        accuracy_query))

    def correct(self, x, N=0, laplace=False):
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        x_train = x_support
        y_train = y_support

        target_list = list()
        samples_per_model = int(len(y_train) / self.n_way)

        z_train = self.feature_extractor.forward(x_train).detach() #[340, 64]
        if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
        train_list = [z_train]*self.n_way

        sample_fn, _ = get_transforms(self.cnf, self.use_conditional)
        for way in range(self.n_way):
            # print('WAY: ', way)
            target = torch.ones(len(y_train), dtype=torch.float32) * -1.0

            start_index = way * samples_per_model
            stop_index = start_index + samples_per_model
            target[start_index:stop_index] = 1.0

            target_list.append(target.unsqueeze(1).cuda())

        y = torch.cat(target_list, axis=1).unsqueeze(1)
        #y = y + torch.randn(y.size()).to(y)
        y, delta_log_py = self.cnf(y, z_train.detach(), torch.zeros(y.size(0), y.size(1), 1).to(y))


        for idx, single_model in enumerate(self.model.models):
            single_model.set_train_data(inputs=z_train, targets=y[:, 0, idx], strict=False)

        optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': 0.001},
                                      {'params': self.cnf.parameters(), 'lr': 0.001}])

        self.model.train()
        self.likelihood.train()
        self.feature_extractor.eval()
        self.cnf.train()

        avg_loss=0.0
        for i in range(0, N):
            ## Optimize
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            loss = -self.mll(output, self.model.train_targets) + torch.sum(delta_log_py)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32):
            self.model.eval()
            self.likelihood.eval()
            self.feature_extractor.eval()
            self.cnf.eval()
            z_query = self.feature_extractor.forward(x_query).detach()
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            z_query_list = [z_query]*len(y_query)
            predictions = self.likelihood(*self.model(*z_query_list))  # return 20 MultiGaussian Distributions
            gaussian_means = []
            for gaussian in predictions:
                gaussian_means.append(gaussian.mean.unsqueeze(1))
            gaussian_means = torch.cat(gaussian_means, axis=1).unsqueeze(1)
            preds = sample_fn(gaussian_means, z_query).squeeze()
            predictions_list = torch.sigmoid(preds).cpu().detach().numpy()
            y_pred = predictions_list.argmax(axis=1)  #[model, classes]
            top1_correct = np.sum(y_pred == y_query)
            count_this = len(y_query)
        return float(top1_correct), count_this, avg_loss/float(N+1e-10)

    def test_loop(self, test_loader, record=None, return_std=False):
        print_freq = 10
        correct =0
        count = 0
        acc_all = []
        iter_num = len(test_loader)
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this, loss_value = self.correct(x)
            acc_all.append(correct_this/ count_this*100)
            if(i % 100==0):
                acc_mean = np.mean(np.asarray(acc_all))
                print('Test | Batch {:d}/{:d} | Loss {:f} | Acc {:f}'.format(i, len(test_loader), loss_value, acc_mean))
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if(self.writer is not None): self.writer.add_scalar('test_accuracy', acc_mean, self.iteration)
        if(return_std): return acc_mean, acc_std
        else: return acc_mean


    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        ##Dividing input x in query and support set
        x_support = x[:,:self.n_support,:,:,:].contiguous().view(self.n_way * (self.n_support), *x.size()[2:]).cuda()
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
        x_query = x[:,self.n_support:,:,:,:].contiguous().view(self.n_way * (self.n_query), *x.size()[2:]).cuda()
        y_query = np.repeat(range(self.n_way), self.n_query)

        # Init to dummy values
        x_train = x_support
        y_train = y_support
        target_list = list()
        samples_per_model = int(len(y_train) / self.n_way)
        for way in range(self.n_way):
            target = torch.ones(len(y_train), dtype=torch.float32) * -1.0
            start_index = way * samples_per_model
            stop_index = start_index+samples_per_model
            target[start_index:stop_index] = 1.0
            target_list.append(target.cuda())
        z_train = self.feature_extractor.forward(x_train).detach() #[340, 64]
        if(self.normalize): z_train = F.normalize(z_train, p=2, dim=1)
        train_list = [z_train]*self.n_way
        for idx, single_model in enumerate(self.model.models):
            single_model.set_train_data(inputs=z_train, targets=target_list[idx], strict=False)


        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32):
            self.model.eval()
            self.likelihood.eval()
            self.feature_extractor.eval()
            z_query = self.feature_extractor.forward(x_query).detach()
            if(self.normalize): z_query = F.normalize(z_query, p=2, dim=1)
            z_query_list = [z_query]*len(y_query)
            predictions = self.likelihood(*self.model(*z_query_list)) #return n_way MultiGaussians
            predictions_list = list()
            for gaussian in predictions:
                predictions_list.append(gaussian.mean) #.cpu().detach().numpy())
            y_pred = torch.stack(predictions_list, 1)
        return y_pred

class ExactGPLayer(gpytorch.models.ExactGP):
    '''
    Parameters learned by the model:
        likelihood.noise_covar.raw_noise
        covar_module.raw_outputscale
        covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, config, train_x, train_y, likelihood, kernel='linear'):
        #Set the likelihood noise and enable/disable learning
        likelihood.noise_covar.raw_noise.requires_grad = False
        likelihood.noise_covar.noise = torch.tensor(0.1)
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        ## Linear kernel
        if(kernel=='linear'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        ## RBF kernel
        elif(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Matern kernel
        elif(kernel=='matern'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        ## Polynomial (p=1)
        elif(kernel=='poli1'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=1))
        ## Polynomial (p=2)
        elif(kernel=='poli2'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        elif(kernel=='cossim' or kernel=='bncossim'):
        ## Cosine distance and BatchNorm Cosine distance
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            self.covar_module.base_kernel.variance = 1.0
            self.covar_module.base_kernel.raw_variance.requires_grad = False
        elif(kernel=="nn"):
            self.kernel = NNKernel(input_dim=config.nn_config["input_dim"],
                              output_dim=config.nn_config["output_dim"],
                              num_layers=config.nn_config["num_layers"],
                              hidden_dim=config.nn_config["hidden_dim"])
            self.covar_module = gpytorch.kernels.ScaleKernel(self.kernel)
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
