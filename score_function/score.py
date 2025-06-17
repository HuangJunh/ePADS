import torch
import numpy as np
import sys
sys.path.append('../score_function')
sys.path.append('./score_function')
from Product import score_pads

def score(network, data, data_noise, device, args):

    pack = (data, data_noise)
    diffs = score_pads(network, pack, device, args)
    try:
        Psi = np.log(diffs.item())
    except Exception as e:
        Psi = 0

    return Psi


def score_nds(network, device, args, data, target, noise):

    data_noise = data + noise
    pack = (data, data_noise)
    diffs = score_pads(network, pack, device, args)
    try:
        Psi = np.log(diffs.item())
    except Exception as e:
        print(e)
        Psi = 0

    return Psi
