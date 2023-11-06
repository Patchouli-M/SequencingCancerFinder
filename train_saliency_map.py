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
import matplotlib.pyplot as plt
import seaborn as sns

# get arguments for training
args = args_utils.get_args()
args.output = args.output.replace('normal','saliency')
os.makedirs(args.output,exist_ok=True)

# get model
args_utils.set_random_seed(args.seed)
args.HVG_list = opt_utils.generate_genelist(args)

# get dataloaders for training
train_loaders = domian_loaders.train_domian_loaders_l(args)
val_loaders = domian_loaders.val_domian_loaders_l(args)

# i/o for log output
f_loss_io = open( os.path.join(args.output,f'{args.logs_name}_loss.txt'),'w')
f_val_io = open( os.path.join(args.output,f'{args.logs_name}_val.txt'),'w')
[print(_,file=f_val_io,end='\t') if idx!=len(val_loaders)-1 else print(_,file=f_val_io,end='\n') for idx, _ in enumerate(val_loaders) ]
epoch_sum_grad = []
algorithm = model.VREx(args)

# train
for epoch in range(args.max_epoch):
    train_minibatches_iterator = zip(*train_loaders)
    count = 0 
    grad_list = []
    for single_train_minibatches in train_minibatches_iterator : 
        for single_domain in single_train_minibatches:
                single_domain[0].requires_grad_()
        count+=1
        algorithm.train()
        if args.gpu_id :
            algorithm.cuda()
        minibatches_device = [(data) for data in single_train_minibatches]      
        opt = opt_utils.get_optimizer(algorithm, args)
        sch = opt_utils.get_scheduler(opt, args)
        # back-propagation
        step_vals = algorithm.update(minibatches_device, opt, sch)
        print(step_vals,file=f_loss_io) 
        for single_domain in single_train_minibatches:
            grad_list.append(single_domain[0].grad.data.abs().numpy())
    algorithm.eval()
    algorithm.cpu()

    # evaluate accuracy during training
    for idx,loader_idx in enumerate(val_loaders):
        acc = opt_utils.accuracy(algorithm,val_loaders[loader_idx])
        if idx!=len(val_loaders)-1:
            print (f'{acc:.4f}',file=f_val_io,end='\t')
        else :
            print (f'{acc:.4f}',file=f_val_io,end='\n')
        print(f'{acc:.4f}',end='\t')
    f_val_io.flush()   
    print(f'epoch={epoch}',end='\n')    
    print(step_vals)
    
    # save the gradient values.
    epoch_total_grad = np.concatenate(grad_list,axis=0).sum(axis=0)
    epoch_sum_grad.append(epoch_total_grad.reshape(-1,epoch_total_grad.shape[0]))
    grad_df = pd.DataFrame(np.concatenate(epoch_sum_grad,axis=0),columns=args.HVG_list)
    grad_df.to_csv(os.path.join(args.output,'grad.csv'))
    grad_df = grad_df.T
    grad_df['mean'] = grad_df.mean(axis=1)
    grad_df = grad_df.sort_values(by='mean',ascending=False)
    grad_df.iloc[:20].loc[:,'mean'].to_csv(os.path.join(args.output,f'top_20_gene_list_{epoch}.txt'),header=None)
    grad_df = grad_df.drop(columns='mean')
    plt.figure(figsize=(10,10))
    # save the figure
    sns.heatmap(grad_df.iloc[:20],cmap='coolwarm')
    plt.savefig(os.path.join(args.output,f'saliency_map.png'),bbox_inches='tight')

f_val_io.close()
f_loss_io.close()




