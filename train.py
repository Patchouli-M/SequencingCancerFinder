# %%
from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets,transforms
import torch
import pandas as pd 
from models import model
from utils import opt_utils,args_utils 
import importlib
import glob
import os
import numpy as np 
from data_loaders import domian_loaders
args = args_utils.get_args()
args_utils.set_random_seed(args.seed)
opt_utils.generate_genelist(args)

# %%
train_loaders = domian_loaders.train_domian_loaders_l(args)
val_loaders = domian_loaders.val_domian_loaders_l(args)
f_loss_io = open( os.path.join(args.output,f'{args.logs_name}_loss.txt'),'w')
f_val_io = open( os.path.join(args.output,f'{args.logs_name}_val.txt'),'w')

# %%
if args.gpu_id :
    algorithm = model.VREx(args).cuda()
else : 
    algorithm = model.VREx(args)
for epoch in range(args.max_epoch):
    train_minibatches_iterator = zip(*train_loaders)
    count = 0 
    for single_train_minibatches in train_minibatches_iterator : 
        count+=1
        algorithm.train()
        minibatches_device = [(data) for data in single_train_minibatches]      
        opt = opt_utils.get_optimizer(algorithm, args)
        sch = opt_utils.get_scheduler(opt, args)
        step_vals = algorithm.update(minibatches_device, opt, sch)
        print(step_vals,file=f_loss_io)
        algorithm.eval()
        acc = opt_utils.accuracy(algorithm,train_loaders[0])
        print (f'{acc:.4f}',file=f_val_io,end='\n')
    opt_utils.save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
f_val_io.close()
f_loss_io.close()


