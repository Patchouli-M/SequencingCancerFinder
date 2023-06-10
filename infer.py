# %%


from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets,transforms
import torch
import pandas as pd 
from models import model
from utils import opt_utils,args_utils
import warnings
import argparse
warnings.filterwarnings('ignore')

# %%
parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckp', type=str, default='checkpoints/sample_ckp.pkl')
parser.add_argument('--HVG', type=str, default='data/test_data/sample_gene_list.txt')
parser.add_argument('--matrix', type=str, default='data/test_data/sample_data_matrix.txt')
parser.add_argument('--out', type=str, default='out.csv')
args_infer = parser.parse_args()
print(args_infer.out)
# %%
checkpoint_file = args_infer.ckp
scRNA_HVG_list = args_infer.HVG
counts_file = args_infer.matrix
input_data = opt_utils.normalize_scRNA_counts(counts_file,scRNA_HVG_list)
input_set = TensorDataset(torch.from_numpy(input_data.values).float())
input_loader = DataLoader(dataset=input_set,batch_size = len(input_set))
args = args_utils.get_args()
args_utils.set_random_seed(args.seed)
algorithm = model.VREx(args)
algorithm.load_state_dict(torch.load(checkpoint_file)['model_dict'])
algorithm.eval()
predict_dict = {'sample':[],'malignant_state':[]}
for data in input_loader:
    out = torch.argmax(algorithm.predict(data[0]),axis=1)
for idx,i in enumerate(out):
    predict_dict['sample'].append(input_data.index[idx])
    predict_dict['malignant_state'].append(i.item())
predict_df = pd.DataFrame(predict_dict)
predict_df.to_csv(args_infer.out,index=False)


