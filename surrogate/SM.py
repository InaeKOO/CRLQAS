import argparse
import torch
from sys import argv

class SM(object):
    def __init__(self, conf, action_size, state_size)

def initialize_model(model):
    pass

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='Set specific GPU to run experiment [0, 1, ...]')
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    device = torch.device(f"cuda:{args.gpu_id}")
