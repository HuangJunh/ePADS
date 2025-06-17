import torch
import numpy as np
from utils import add_dropout, init_network

def score_pads(network, pack, device, args):
    network = network.cuda()
    try:
        if args.dropout:
            add_dropout(network, args.sigma)
        if args.init != '':
            init_network(network, args.init)

        network.diffs = 0

        def counting_forward_hook_act(module, inp, out):
            try:
                if isinstance(inp, tuple):
                    inp = inp[0]

                feature, feature_noise = torch.split(inp, [args.batch_size, args.batch_size], 0)
                xx = torch.gt(feature, 0).float()
                xx_shuffle = xx[torch.randperm(xx.size(0))]
                xx_noise = torch.gt(feature_noise, 0).float()
                diff1 = torch.sum(torch.abs(xx_shuffle - xx))
                diff2 = torch.sum(torch.abs(xx_noise - xx))
                network.diffs += diff1 * diff2

            except Exception as e:
                print(e)
                pass

        def counting_backward_hook_act(module, inp, out):
            module.visited_backwards_act = True


        for name, module in network.named_modules():
            if 'ReLU' in str(type(module)):
                module.register_forward_hook(counting_forward_hook_act)

        x, x_noise = pack
        x2 = torch.cat([x, x_noise], 0)

        network(x2)
        diffs = network.diffs

        return diffs

    except Exception as e:
        return np.nan

