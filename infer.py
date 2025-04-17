from torch.utils.data import TensorDataset,DataLoader
import torch
import pandas as pd 
from models import model
from utils import opt_utils,args_utils
import warnings

warnings.filterwarnings('ignore')

# get arguments for inference
args = args_utils.infer_args()
# get model
algorithm = model.VREx(args)
algorithm.load_state_dict(torch.load(args.ckp)['model_dict'])
algorithm.eval()

# get dataloaders for inference
infor_loaders = opt_utils.InferLoaders(args)
predict_dict_res_l = []
predict_dict = {'sample':[],'predict':[]}
count = 0

# get data from dataloaders
for input_data,input_loader in infor_loaders:
    print(f'begin {count}')
    count+=1
    # perdict
    for data in input_loader:
        out = torch.softmax(algorithm.predict(data[0]),axis=1)[:,1]
    for idx,i in enumerate(out):
        predict_dict['sample'].append(input_data.index[idx])
        predict_dict['predict'].append(i.item())
# generate dataframe from results
predict_df = pd.DataFrame(predict_dict)
predict_df['predict'][predict_df['predict'] > args.threshold] = 1
predict_df['predict'][predict_df['predict'] != 1 ] = 0

# save results
predict_df.to_csv(args.out,index=False)
print(predict_df)


