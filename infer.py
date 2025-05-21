from torch.utils.data import TensorDataset,DataLoader
import torch
import pandas as pd 
from models import model
from utils import opt_utils,args_utils
import warnings
from typing import Optional

def run_inference(args):
    """
    Run inference using the provided arguments.
    
    Args:
        args: Arguments for inference.
    
    Returns:
        DataFrame containing the inference results.
    """
    # Initialize model with args
    algorithm = model.VREx(args)
    algorithm.load_state_dict(torch.load(args.ckp)['model_dict'])
    algorithm.eval()

    # get dataloaders for inference
    infor_loaders = opt_utils.InferLoaders(args)
    predict_dict = {'sample':[],'predict':[]}
    count = 0

    # get data from dataloaders
    for input_data,input_loader in infor_loaders:
        print(f'begin {count}')
        count+=1
        # predict
        for data in input_loader:
            out = torch.softmax(algorithm.predict(data[0]),axis=1)[:,1]
        for idx,i in enumerate(out):
            predict_dict['sample'].append(input_data.index[idx])
            predict_dict['predict'].append(i.item())
    # generate dataframe from results
    predict_df = pd.DataFrame(predict_dict)
    predict_df['predict'][predict_df['predict'] > args.threshold] = 1
    predict_df['predict'][predict_df['predict'] != 1 ] = 0
    
    return predict_df

def infering(
    # Model parameters
    matrix = 'data_matrix.h5ad',
    num_classes: int = 2,
    ckp: str = 'checkpoints/sc_pretrain_article.pkl',
    threshold: float = 0.5,
    out: str = 'out.csv',
    
    # Training parameters (kept for model initialization)
    lr: float = 1e-3,
    schuse: bool = False,
    schusech: str = 'cos',
    seed: int = 0,
    gpu_id: Optional[int] = None,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    lr_decay: float = 0.75,
    lr_decay1: float = 1.0,
    lr_decay2: float = 1.0,
    anneal_iters: int = 500,
    max_epoch: int = 5,
    lam: float = 1,
    gene_num: int = 4572,
    label_str: str = 'label',
    batch_size: int = 10,
    NEED_ROWS: int = 50,
    save_output: bool = True
) -> None:
    """
    Perform inference using a pre-trained model on AnnData object.
    
    Args:
        num_classes (int): Number of classes for classification
        ckp (str): Path to the checkpoint file containing the pre-trained model
        threshold (float): Classification threshold for binary predictions
        out (str): Path to save the output predictions
        lr (float): Learning rate
        schuse (bool): Whether to use scheduler
        schusech (str): Scheduler type
        seed (int): Random seed
        gpu_id (Optional[int]): GPU device ID
        weight_decay (float): Weight decay for optimizer
        momentum (float): Momentum for optimizer
        lr_decay (float): Learning rate decay
        lr_decay1 (float): Learning rate decay for pretrained featurizer
        lr_decay2 (float): Initial learning rate decay of network
        anneal_iters (int): Penalty anneal iterations used in VREx
        max_epoch (int): Maximum number of epochs
        lam (float): Tradeoff hyperparameter used in VREx
        gene_num (int): Number of features
        label_str (str): Row label for the label in training data
        batch_size (int): Batch size
        NEED_ROWS (int): Number of samples in one step
        save_output (bool): Option to save or not the output
    
    Returns:
        pd.DataFrame
    """
    # Convert function arguments to args object
    args = args_utils.create_args_from_infering(
        matrix=matrix,
        num_classes=num_classes,
        ckp=ckp,
        threshold=threshold,
        out=out,
        lr=lr,
        schuse=schuse,
        schusech=schusech,
        seed=seed,
        gpu_id=gpu_id,
        weight_decay=weight_decay,
        momentum=momentum,
        lr_decay=lr_decay,
        lr_decay1=lr_decay1,
        lr_decay2=lr_decay2,
        anneal_iters=anneal_iters,
        max_epoch=max_epoch,
        lam=lam,
        gene_num=gene_num,
        label_str=label_str,
        batch_size=batch_size,
        NEED_ROWS=NEED_ROWS
    )
    
    predict_df = run_inference(args)

    # save results
    if save_output == True:
        predict_df.to_csv(args.out,index=False)
        
    return predict_df



warnings.filterwarnings('ignore')

if __name__=='__main__':
    # get arguments for inference
    args = args_utils.infer_args()
    # Run inference
    predict_df = run_inference(args)

    print(predict_df)
    predict_df.to_csv(args.out)


