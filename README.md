
## Set Up Environment

Prerequisites:
```
System: Ubuntu 18.04
python: 3.9.16
CUDA: 11.6
torch: 1.13.1
```
Install environment :
```
conda create -n scf python==3.9.16
conda activate scf
pip install -r requirements.txt
```

## Running the Code
### Usage and Options:
Usage for **inference**:
```
python -u infer.py \
    --ckp=<ckp_file> \
    --HVG=<HVG_file> \
    --matrix=<data_file> \
    --out=<output_file>
```
Options for **inference**:
```
  --ckp     checkpoint file of pretrained model
  --HVG     generated by training, an example is sample_data/sample_gene_list.txt
  --matrix  input data for which predictions need to be made, an example is sample_data/sample_data_matrix.txt
  --out     output file
```

the [HVG](https://drive.google.com/file/d/1BVllGyh2DDbtmzmRS95C6k83nyaZ8YoA) and [checkpoint](https://drive.google.com/file/d/1B5upgf0FT9d-jsji_vdv5jxz8oybL1On) trained by us  are available for download.   
If you want to use our pretrained model for inference, please:  
1. Download the HVG and checkpoint above and specify them with the parameter `--HVG` and `--ckp`.
2. Specify the `count_file`  you need with the parameter `--matrix`.  

an inference sample as :

```
python -u infer.py --ckp=checkpoints/sample_ckp.pkl --HVG=sample_data/sample_gene_list.txt --matrix=sample_data/sample_data_matrix.txt --out=out.csv
```
---
If you want to **train** your dataset, please use:
```
python -u train.py  \
    --train_dir=<train_dir> \     # Directory of training data
    --val_dir=<val_dir> \         # Directory of val data
    --batch_size=<batch_size> \   # batch size
    --lr=<learning_rate> \        # learning rate
    --max_epoch=<max_epoch> \     # max epoch
    --output=<output_dir> \       # The output directory of training, the output of training mainly including checkpoint and HVG, and logs files
    --gpu_id=<id>                 # Not necessary, Specify the No. of the gpu if it is available
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.