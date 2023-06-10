import torch
import os
import pandas as pd 
import scanpy
import anndata
import glob


def get_params(alg, args, inner=False, alias=True, isteacher=False):
    if args.schuse:
        if args.schusech == 'cos':
            initlr = args.lr
        else:
            initlr = 1.0
    else:
        if inner:
            initlr = args.inner_lr
        else:
            initlr = args.lr
    if isteacher:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 * initlr},
            {'params': alg[2].parameters(), 'lr': args.lr_decay2 * initlr}
        ]
        return params
    if inner:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 *
             initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 *
             initlr}
        ]
    elif alias:
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * initlr}
        ]
    else:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 * initlr}
        ]
    return params

def get_optimizer(alg, args, inner=False, alias=True, isteacher=False):
    params = get_params(alg, args, inner, alias, isteacher)
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    return optimizer

def get_scheduler(optimizer, args):
    if not args.schuse:
        return None
    if args.schusech == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch * args.steps_per_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cpu().float()
            y = data[1].cpu().long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total

def accuracy_norm(network, loader,minmax0,minmax1):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict_norm(x,minmax0,minmax1)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total


def save_checkpoint(filename, alg, args):
    save_dict = {
        "args": vars(args),
        "model_dict": alg.cpu().state_dict()
    }
    torch.save(save_dict, os.path.join(args.output, filename))

def normalize_scRNA_counts(count_file,HVG_file,HVG_num = 4096,target_sum=10000):
    TRAIN_GENE_LIST = HVG_file
    gene_list = pd.read_csv(TRAIN_GENE_LIST,header=None,index_col=0).iloc[:HVG_num]
    raw_df = pd.read_csv(count_file,index_col=0,sep='\t')
    raw_df['sum'] = raw_df.sum(axis=1)
    raw_df = raw_df.sort_values(by='sum',ascending=False)
    raw_df = raw_df.drop(columns='sum')
    raw_df = raw_df[~raw_df.index.duplicated()]
    raw_df = raw_df.T
    adata = anndata.AnnData(raw_df,raw_df.index.to_frame(), raw_df.columns.to_frame())
    scanpy.pp.normalize_total(adata,target_sum=target_sum)
    scanpy.pp.log1p(adata)
    raw_df = pd.DataFrame(adata.X,index=adata.obs.index,columns=adata.var.index)
    select_df = pd.merge(gene_list,raw_df.T,how='left',left_index=True,right_index=True)
    select_df = select_df.fillna(0.0).T[gene_list.index]
    select_df = select_df.sort_index()
    return select_df


def generate_genelist(args):
    fst_flag = True
    merged_df = pd.DataFrame()
    for file_name in glob.glob(os.path.join(args.train_dir,'*')):
        raw_df = pd.read_csv(file_name,sep='\t',index_col=0)
        raw_df['sum'] = raw_df.sum(axis=1)
        raw_df = raw_df.sort_values(by='sum',ascending=False)
        raw_df = raw_df.drop(columns='sum')
        raw_df = raw_df[~raw_df.index.duplicated()]
        
        if fst_flag:
            merged_df = raw_df.copy()
        else :
            merged_df = pd.merge(merged_df,raw_df,left_index=True,right_index=True,how='inner')
    merged_df = merged_df[merged_df.index!=args.label_str]
    merged_df['var'] = merged_df.var(axis=1)
    merged_df = merged_df.sort_values(by='var',ascending=False)
    genelist = list(merged_df.index[:args.gene_num])
    genelist.append(args.label_str)
    pd.DataFrame(index=genelist).to_csv(os.path.join(args.output,args.genelist_outname),header=None)
    return os.path.join(args.output,args.genelist_outname)