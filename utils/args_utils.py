import os
import sys
import time
import numpy as np
import argparse
import torch
import random
import torchvision
import PIL

def get_time_str():
    return time.strftime('%m%d-%H%M')

def get_args():
    date_str = time.strftime('%m%d')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_dir', type=str, default='data/train', help="train_dir")
    parser.add_argument('--val_dir', type=str, default='data/val', help="val_dir")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=None, help="device id to run")
    parser.add_argument('--num_classes', type=int, default=2, help="num classes")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float,default=0.9, help='for optimizer')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float, default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,help='inital learning rate decay of network')
    parser.add_argument('--anneal_iters', type=int,default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--max_epoch', type=int,default=10, help="max iterations")
    parser.add_argument('--output', type=str,default=f'normal_train_{ date_str }', help="output dir")
    parser.add_argument('--genelist_outname', type=str,default='gene_list.txt', help="genelist_outname")
    parser.add_argument('--logs_name', type=str,default='train_log', help="train_log")
    parser.add_argument('--lam', type=float,default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--gene_num', type=int,default=4096, help="num of HVG")
    parser.add_argument('--label_str', type=str,default='label', help="num of HVG")
    parser.add_argument('--batch_size', type=int,default=10, help="batch_size")
    parser.add_argument('--NEED_ROWS', type=int,default=50, help="NEED_ROWS")
    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    os.makedirs(args.output,exist_ok=True)
    return args


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

