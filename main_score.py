import argparse
import datasets
import random
import numpy as np
import torch
import os, copy
from scipy import stats
from pycls.models.nas.nas import Cell
from searchspace import searchspace
import datasets.data as data
from datasets.perturbation_data.perturbation_data import *
from score_function.score import score_nds

parser = argparse.ArgumentParser(description='ePADS')
parser.add_argument('--data_loc', default='./datasets/CIFAR10_data/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='./APIs/NAS-Bench-201-v1_1-096897.pth', type=str, help='path to API')
parser.add_argument('--save_loc', default='./results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='ePADS', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--ptype', default='nds_darts', type=str, help='the nas search space to use, nds_pnas nds_enas  nds_darts nds_darts_fix-w-d nds_nasnet nds_amoeba nds_resnet nds_resnext-a nds_resnext-b')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=1, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='1', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--n_samples', default=1000, type=int)
parser.add_argument('--n_runs', default=100, type=int)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
savedataset = args.dataset
dataset = 'fake' if 'fake' in args.dataset else args.dataset
args.dataset = args.dataset.replace('fake', '')
searchspace = searchspace.get_search_space(args)
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)

filename = f'{args.save_loc}/{args.save_string}_{args.ptype}_{savedataset}{"_" + args.init + "_" if args.init != "" else args.init}_{"_dropout" if args.dropout else ""}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}'
accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.ptype}_{savedataset}_{args.trainval}'

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

scores = np.zeros(len(searchspace))
try:
    accs = np.load(accfilename + '.npy')
except:
    accs = np.zeros(len(searchspace))
print('Start!!!')
print("sigma={}".format(args.sigma))
g = GaussianNoise(train_loader, device, sigma=args.sigma)
data, target, noise = g.get_noise_data()
for i, (uid, network) in enumerate(searchspace):
    # Reproducibility
    try:
        s = score_nds(network, device, args, data, target, noise)

        scores[i] = s
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        accs_ = accs[~np.isnan(scores)]
        scores_ = scores[~np.isnan(scores)]
        numnan = np.isnan(scores).sum()
        # if i % 100 == 0:
        tau, p = stats.kendalltau(accs_[:max(i - numnan, 1)], scores_[:max(i - numnan, 1)])
        print('search_space: ', str(args.ptype), 'kendall_tau:', f'{tau}')
        if i % 1000 == 0:
            np.save(filename, scores)
            np.save(accfilename, accs)
    except Exception as e:
        print(e)
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        scores[i] = np.nan
np.save(filename, scores)
np.save(accfilename, accs)
print("sigma={}".format(args.sigma))
tau, p = stats.kendalltau(accs, scores)
rho, pp = stats.spearmanr(accs, scores)
print("tau={}, rho={}".format(tau, rho))
print('Finished!!!')