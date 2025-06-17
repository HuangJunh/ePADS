import torch
import numpy as np
import random
import pandas as pd
import argparse
import importlib
from statistics import mean, stdev
import re, os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seed',default=None,type=int,help='random seed')
parser.add_argument('--algo',default='rs',type=str,help='algorithm (rs, se)')
parser.add_argument('--sigma',default=1,type=float,help='standard deviation of gaussian noise')
parser.add_argument('--encoding',default='backbone',type=str,help='encoding scheme (hash,backbone)')
parser.add_argument('--runs',default=50,type=int,help='number of runs')
parser.add_argument('--iters',default=1000,type=int,help='number of iterations for each run')
parser.add_argument('--sl',default=21,type=int,help='sequence length')
parser.add_argument('--ptype',default='nasbench201',type=str,help='problem type (nasbench101,nasbench201,etc.)')
parser.add_argument('--atom',default=None,type=int,help='number of choices per node/edge')
parser.add_argument('--max_evaluations',default=1000,type=int,help='number of evaluations for each run')
parser.add_argument('--GPU', default='0', type=str)
#----------------------------------------------------SE parameters--------------------------------------------------------#
parser.add_argument('--n',default=4,type=int,help='number of searchers')
parser.add_argument('--h',default=4,type=int,help='number of regions (SE)')
parser.add_argument('--w',default=2,type=int,help='the number of possible goods (the number of samples of each region)  (SE)')
#----------------------------------------------------NAS-BENCH-201 & NATS-BENCH-SSS---------------------------------------#
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_loc', default='./datasets/CIFAR10_data/', type=str, help='dataset folder')
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--init', default='', type=str)
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--presample', default=20, type=int, help='pre-sample size')
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
if args.seed is not None:
    torch.manual_seed(args.seed)


from score_function.score import score
from searchspace import searchspace
import datasets.data as data
from searchspace.nas_201_encoding import BACKBONE as ENCODING

encoding = ENCODING() 

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

train_loader = data.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
algo_module = importlib.import_module(f'{args.algo}.{args.algo}_{args.ptype}')
init = getattr(algo_module, args.algo)
ss = getattr(searchspace, args.ptype.upper())
ss = ss(args.dataset, args)

# if re.search('se[a-z]*', args.algo):
#     par = {'iters':args.iters, 'n':args.n, 'h':args.h, 'w':args.w, 'sl':args.sl, 'atom':args.atom, 'max_evaluations':args.max_evaluations}
if re.search('rs[a-z]*', args.algo):
    par = {'sample': args.max_evaluations}
else:
    raise("No such algo!")

    
if args.ptype=='nasbench201':
    get_acc = lambda id:ss.get_acc_by_code(encoding.parse_code(id),args)
    get_acc_all = lambda id:ss.get_acc_by_code_all(encoding.parse_code(id),args)
    get_acc_proxy = lambda id:ss.get_acc_by_code(encoding.parse_code(id),args,hp=args.hp)
    get_time_proxy = lambda id:ss.get_training_time_by_code(encoding.parse_code(id),args,hp=args.hp)
    get_net = lambda id:ss.get_net_by_code(encoding.parse_code(id),args)


if args.dataset == 'cifar10':
    args.acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    args.acc_type = 'x-test'
    val_acc_type = 'x-valid'

hist_code = []
hist_gbest = []
hist_runtime = []
hist_trainingtime = []
hist_acc = []
hist_valid = []
hist_acc_cifar10 = []
hist_valid_cifar10 = []
hist_acc_cifar100 = []
hist_valid_cifar100 = []
hist_acc_imagenet = []
hist_valid_imagenet = []

print("Problem type: {}".format(args.ptype))
print("Algorithm: {}".format(args.algo))
print("Number of evaluations for each run: {}".format(args.max_evaluations))

if args.algo=="rs":
    par['ss'] = ss

for r in range(args.runs):
    start = time.time()

    data_iterator = iter(train_loader)
    data, target = next(data_iterator)
    noise = (data.new(data.size()).normal_(0, args.sigma))
    data_noise = data + noise
    ff = lambda code:score(get_net(code), data.to(device), data_noise.to(device), device, args) # function ff(code), return score

    # avoid repeatedly scoring same networks
    dictionary = {}
    par['dictionary'] = dictionary
    def dff(code):
        index = tuple(code)
        if index in dictionary:
            return dictionary[index]
        else:
            r = ff(code)
            dictionary[index] = r
            return r
    par['ff'] = dff
    
    # Initialize search algorithm
    algo = init(**par)
    
    gbest_code, gbest = algo.Search()
    end = time.time()
    training_time=0 # For non-training-free only

    hist_code.append(gbest_code)
    hist_gbest.append(gbest)
    hist_runtime.append(end-start)
    hist_trainingtime.append(training_time)

    if args.ptype=='nasbench201':
        acc_cifar10, valid_cifar10, acc_cifar100, valid_cifar100, acc_imagenet, valid_imagenet = get_acc_all(gbest_code)
        print("Run {:3d}: gbest = {:.3f};acc1 = {:.2f};valid1 = {:.2f};acc2 = {:.2f};valid2 = {:.2f};acc3 = {:.2f};valid3 = {:.2f}; run time = {:.2f}s;training time = {:.2f}; code = {}".format(r,gbest,acc_cifar10,valid_cifar10,acc_cifar100,valid_cifar100,acc_imagenet,valid_imagenet,end-start,training_time,gbest_code))
        df = pd.DataFrame([[r,gbest,acc_cifar10,valid_cifar10,acc_cifar100,valid_cifar100,acc_imagenet,valid_imagenet,end-start,training_time,gbest_code]],columns=['Run','gbest','cifar10_acc','cifar10_valid','cifar100_acc','cifar100_valid','imagenet_acc','imagenet_valid','run time','training time','code'])
    
        hist_acc_cifar10.append(acc_cifar10)
        hist_valid_cifar10.append(valid_cifar10)
        hist_acc_cifar100.append(acc_cifar100)
        hist_valid_cifar100.append(valid_cifar100)
        hist_acc_imagenet.append(acc_imagenet)
        hist_valid_imagenet.append(valid_imagenet)
        print("Average over {} runs: gbest = {:.3f};acc1 = {:.2f};valid1 = {:.2f};acc2 = {:.2f};valid2 = {:.2f};acc3 = {:.2f};valid3 = {:.2f}; run time = {:.2f}s; training time = {:.2f}s".format(r,mean(hist_gbest),mean(hist_acc_cifar10),mean(hist_valid_cifar10),mean(hist_acc_cifar100),mean(hist_valid_cifar100),mean(hist_acc_imagenet),mean(hist_valid_imagenet),mean(hist_runtime),mean(hist_trainingtime)))

          
print("Average gbest over {} runs: {}".format(args.runs,mean(hist_gbest)))
print("Average run time over {} runs: {:.2f}s".format(args.runs,mean(hist_runtime)))
print("Maximum gbest over {} runs: {}".format(args.runs,max(hist_gbest)))
print("Minimum gbest over {} runs: {}".format(args.runs,min(hist_gbest)))
print("sigma={}".format(args.sigma))
if args.ptype=='nasbench201':
    print("CIFAR-10-test: {} runs, mean: {:.2f}, std: {:.2f}".format(args.runs,mean(hist_acc_cifar10),stdev(hist_acc_cifar10)))
    print("CIFAR-10-valid: {} runs, mean: {:.2f}, std: {:.2f}".format(args.runs,mean(hist_valid_cifar10),stdev(hist_valid_cifar10)))
    print("CIFAR-100-test: {} runs, mean: {:.2f}, std: {:.2f}".format(args.runs,mean(hist_acc_cifar100),stdev(hist_acc_cifar100)))
    print("CIFAR-100-valid: {} runs, mean: {:.2f}, std: {:.2f}".format(args.runs,mean(hist_valid_cifar100),stdev(hist_valid_cifar100)))
    print("ImageNet-16-120-test: {} runs, mean: {:.2f}, std: {:.2f}".format(args.runs,mean(hist_acc_imagenet),stdev(hist_acc_imagenet)))
    print("ImageNet-16-120-valid: {} runs, mean: {:.2f}, std: {:.2f}".format(args.runs,mean(hist_valid_imagenet),stdev(hist_valid_imagenet)))
