from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets,transforms
import torch
import pandas as pd 
from models import model
from utils import opt_utils,args_utils
import warnings
import argparse
warnings.filterwarnings('ignore')


args = args_utils.infer_args()

input_data = opt_utils.normalize_matrix_counts(args.matrix,args.HVG)
input_set = TensorDataset(torch.from_numpy(input_data.values).float())
input_loader = DataLoader(dataset=input_set,batch_size = len(input_set))
algorithm = model.VREx(args)
algorithm.load_state_dict(torch.load(args.ckp)['model_dict'])
algorithm.eval()
predict_dict = {'sample':[],'predict':[]}
for data in input_loader:
    out = torch.argmax(algorithm.predict(data[0]),axis=1)
for idx,i in enumerate(out):
    predict_dict['sample'].append(input_data.index[idx])
    predict_dict['predict'].append(i.item())
predict_df = pd.DataFrame(predict_dict)
predict_df.to_csv(args.out,index=False)
print(predict_df)

