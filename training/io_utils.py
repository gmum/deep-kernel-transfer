import argparse
import glob
import os

import numpy as np

from models import backbone

NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "scale", "concatscale"]

model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='few-shot script classification')
    parser.add_argument('--seed', default=0, type=int, help='Seed for Numpy and pyTorch. Default: 0 (None)')
    parser.add_argument('--dataset', default='CUB', help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model', default='Conv4',
                        help='model: Conv{4|6} / ResNet{10|18|34|50|101}')  # 50 and 101 are not used in the paper
    parser.add_argument('--method', default='baseline',
                        help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}')  # relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way', default=5, type=int,
                        help='class num to classify for training')  # baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way', default=5, type=int,
                        help='class num to classify for testing (validation) ')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ')  # still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--kernel-type', type=str, default='rbf',
                        choices=['rbf', 'bncossim', 'matern', 'poli1', 'poli2', 'cossim', 'nn'])
    parser.add_argument('--save_dir', type=str, default='./save/classification')
    parser.add_argument('--neptune', action="store_true", help="whether to use neptune logging.")

    parser.add_argument('--num_classes', default=200, type=int,
                        help='total number of classes in softmax, only used in baseline')  # make it larger than the maximum label value in base class
    parser.add_argument('--save_freq', default=50, type=int, help='Save frequency')
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
    parser.add_argument('--stop_epoch', default=-1, type=int,
                        help='Stopping epoch')  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
    parser.add_argument('--resume', action='store_true',
                        help='continue from previous trained model with largest epoch')
    parser.add_argument('--warmup', action='store_true',
                        help='continue from baseline, neglected if resume is true')  # never used in the paper

    # save features and test
    parser.add_argument('--split', default='novel',
                        help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
    parser.add_argument('--save_iter', default=-1, type=int,
                        help='saved feature from the model trained in x epoch, use the best model if x is -1')
    # only test
    parser.add_argument('--test', action="store_true", help="perform evaluation")
    parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')
    parser.add_argument('--repeat', default=5, type=int,
                        help='Repeat the test N times with different seeds and take the mean. The seeds range is [seed, seed+repeat]')

    return parser.parse_args()


def parse_args_regression():
    parser = argparse.ArgumentParser(description='few-shot script regression')

    parser.add_argument('--seed', default=0, type=int, help='Seed for Numpy and pyTorch. Default: 0 (None)')
    parser.add_argument('--model', default='Conv3', choices=["Conv3", "MLP2"], help='model: Conv{3} / MLP{2}')
    parser.add_argument('--method', default='DKT', help='DKT / transfer')
    parser.add_argument('--dataset', default='QMUL', help='QMUL / sines')
    parser.add_argument('--update_batch_size', default=5, type=int,
                        help='Number of examples used for inner gradient update (K for K-shot learning).')
    parser.add_argument('--meta_batch_size', default=5, type=int, help='Number of tasks sampled per meta-update')
    parser.add_argument('--output_dim', default=1, type=int, help='Input/output dim for generated dataset')
    parser.add_argument('--multidimensional_amp', default=False, type=str2bool,
                        help='Different amplitudes per each example')
    parser.add_argument('--multidimensional_phase', default=True, type=str2bool,
                        help='Different phases per each example')
    parser.add_argument('--noise', default="gaussian", type=str, choices=["gaussian", "heterogeneous", "None", "hetero_multi"],
                        help='Different phases per each example')
    parser.add_argument('--kernel_type', type=str, default='rbf',
                        choices=['rbf', 'spectral', 'bncossim', 'matern', 'poli1', 'poli2', 'cossim', 'nn'])
    parser.add_argument('--save_dir', type=str, default='./save/regression')
    parser.add_argument('--num_tasks', type=int, default=1, help="the dimension of the target.")
    parser.add_argument('--multi_type', type=int, choices=[2,3], default=3, help="type of nn multi-kernel, used if num-tasks>1 "
                                                                      "and kernel type == n")
    parser.add_argument('--method_lr', type=float, default=0.01)
    parser.add_argument('--feature_extractor_lr', type=float, default=0.01)
    parser.add_argument('--cnf_lr', type=float, default=0.001)

    parser.add_argument('--all_lr', type=float, help="if not None, sets up the same given learning rate for all "
                                                     "the parameters.")

    # neptune logging
    parser.add_argument('--neptune', action="store_true", help="whether to use neptune logging.")
    parser.add_argument('--flow', action="store_true", help="whether to use flow approach.")

    parser.add_argument("--use_conditional", default=False, type=str2bool,
                        help='If CNF should be conditional')

    #parser.add_argument("--context_dim", type=int, default=16, help='Dimensionality of the context.')

    parser.add_argument("--context_type", type=str, default='nn', choices=['nn', 'backbone'])

    parser.add_argument(
        "--layer_type", type=str, default="concatsquash",
        choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
    )
    parser.add_argument('--dims', type=str, default='32-32')
    parser.add_argument("--num_blocks", type=int, default=2, help='Number of stacked CNFs.')
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=False)
    parser.add_argument('--add_noise', type=eval, default=False)
    parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
    parser.add_argument("--nonlinearity", type=str, default="tanh", choices=NONLINEARITIES)

    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

    parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
    parser.add_argument('--test_atol', type=float, default=None)
    parser.add_argument('--test_rtol', type=float, default=None)

    parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
    parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)

    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Track quantities
    parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
    parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
    parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
    parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
    parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
    parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

    # train
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
    parser.add_argument('--stop_epoch', default=100, type=int,
                        help='Stopping epoch')  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py

    # test
    parser.add_argument('--test', action="store_true", help="whether to perform only evaluation.")
    parser.add_argument('--n_support', default=5, type=int,
                        help='Number of points on trajectory to be given as support points')
    parser.add_argument('--n_test_epochs', default=10, type=int, help='How many test people?')
    parser.add_argument('--out_of_range', action="store_true", help="whether to perform test also on out of range "
                                                                    "samples. WARNING: FOR NOW IMPLEMENTED ONLY FOR "
                                                                    "SINES")
    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
