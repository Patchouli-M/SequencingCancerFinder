import os
import sys
import time
import numpy as np
import argparse
import torch
import random

def get_time_str():
    """
    Get current time
    
    Args:
        None
    Return:
        a str, MMDD-HHMM
    """
    return time.strftime('%m%d-%H%M')

def get_args():
    """
    Get the arguments for training to run the script and store them in the args object.
    
    Args:
        None
    Return:
        args object with arguments
    """
    date_str = time.strftime('%m%d')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_dir', type=str, default='data/train', help="path for training data")
    parser.add_argument('--val_dir', type=str, default='data/val', help="path for validating data")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=None, help="device id to run")
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float,default=0.9, help='for optimizer')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float, default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,help='inital learning rate decay of network')
    parser.add_argument('--anneal_iters', type=int,default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--max_epoch', type=int,default=5, help="max iterations")
    parser.add_argument('--output', type=str,default=f'normal_log_{ date_str }', help="output dir")
    parser.add_argument('--genelist_outname', type=str,default='gene_list.txt', help="genelist_outname")
    parser.add_argument('--logs_name', type=str,default='train_log', help="train_log")
    parser.add_argument('--lam', type=float,default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--gene_num', type=int,default=4572, help="number of features")
    parser.add_argument('--label_str', type=str,default='label', help="the row label for the label in the training data")
    parser.add_argument('--batch_size', type=int,default=10, help="batch_size")
    parser.add_argument('--NEED_ROWS', type=int,default=50, help="Number of samples in one step")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    return args

def infer_args():
    """
    Get the arguments for inference to run the script and store them in the args object.
    
    Args:
        None
    Return:
        args object with arguments
    """
    date_str = time.strftime('%m%d')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--ckp', type=str, default='checkpoints/sc_pretrain_article.pkl',help="path for pretrained model")
    parser.add_argument('--matrix', type=str, default='data_matrix.tsv',help="path for data(tsv)")
    parser.add_argument('--threshold', type=float, default=0.5,help="threshold of inference")
    parser.add_argument('--out', type=str, default='out.csv',help="out path")
    args = parser.parse_args()
    args.HVG_list = torch.load(args.ckp)['HVG_list']
    return args


def set_random_seed(seed=0):
    """
    set the random seeds
    
    Args:
        seed (int): random seed
    Return:
        None
    """
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_environ():
    """
    Print the current environment and package version

    Returns:
        None
    """
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))



