import pandas as pd 
import torch
import os
import glob
import numpy as np 
from torch.utils.data import TensorDataset,DataLoader
def train_domian_loaders_l(args,shuffle_state=True):
    SPL_PATH = args.train_dir
    NEED_ROWS = args.NEED_ROWS
    batch_size = args.batch_size
    TRAIN_GENE_LIST = os.path.join(args.output,args.genelist_outname)
    need_col_num=args.gene_num
    label_str = args.label_str
    gene_list = pd.read_csv(TRAIN_GENE_LIST,header=None,index_col=0)
    train_loader_l = []
    for data_spl_file in sorted(glob.glob(os.path.join(SPL_PATH,'*'))):
        raw_df = pd.read_csv(data_spl_file,sep='\t',index_col=0)
        raw_df = pd.merge(gene_list,raw_df,how='left',left_index=True,right_index=True).fillna(0)
        raw_df = raw_df[~raw_df.index.duplicated()]
        raw_df = raw_df.loc[gene_list.index]
        raw_df = raw_df.T
        if NEED_ROWS > raw_df.shape[0] :
            raw_df = pd.DataFrame(np.repeat(raw_df.values,NEED_ROWS/(raw_df.shape[0]),axis=0),columns=raw_df.columns)
        else :
            raw_df = raw_df.sample(NEED_ROWS,random_state=42)
        X = raw_df.iloc[:,:need_col_num].to_numpy()
        Y = raw_df.loc[:,label_str].to_numpy()
        raw_set = TensorDataset(torch.from_numpy(X).float(),torch.from_numpy(Y).float())
        train_set = raw_set
        train_loader = DataLoader(dataset=train_set,batch_size = batch_size,shuffle=shuffle_state,drop_last=True)
        train_loader_l.append(train_loader)
    return train_loader_l

def val_domian_loaders_l(args,shuffle_state=True):
    SPL_PATH = args.val_dir
    NEED_ROWS = args.NEED_ROWS
    batch_size = args.batch_size
    TRAIN_GENE_LIST = os.path.join(args.output,args.genelist_outname)
    need_col_num=args.gene_num
    label_str = args.label_str
    gene_list = pd.read_csv(TRAIN_GENE_LIST,header=None,index_col=0)
    train_loader_l = []
    for data_spl_file in sorted(glob.glob(os.path.join(SPL_PATH,'*'))):
        raw_df = pd.read_csv(data_spl_file,sep='\t',index_col=0)
        raw_df = pd.merge(gene_list,raw_df,how='left',left_index=True,right_index=True).fillna(0)
        raw_df = raw_df[~raw_df.index.duplicated()]
        raw_df = raw_df.loc[gene_list.index]
        raw_df = raw_df.T
        if NEED_ROWS > raw_df.shape[0] :
            raw_df = pd.DataFrame(np.repeat(raw_df.values,NEED_ROWS/(raw_df.shape[0]),axis=0),columns=raw_df.columns)
        else :
            raw_df = raw_df.sample(NEED_ROWS,random_state=42)
        X = raw_df.iloc[:,:need_col_num].to_numpy()
        Y = raw_df.loc[:,label_str].to_numpy()
        raw_set = TensorDataset(torch.from_numpy(X).float(),torch.from_numpy(Y).float())
        train_set = raw_set
        train_loader = DataLoader(dataset=train_set,batch_size = batch_size,shuffle=shuffle_state,drop_last=True)
        train_loader_l.append(train_loader)
    return train_loader_l