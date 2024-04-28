# MTSF-DG
Here is the code and appendix for the paper 'Multiple Time Series Forecasting with Dynamic Graph Modeling'.

The data process and data loader are under '/lib/', and the MTSF-DG method is under '/model/'.

## Datasets
Download datasets from 'https://github.com/zhkai/MTSF-DG/releases'.

Move train/test/val to '/data/METR-LA/'.

## Environment 
-  Python                    3.6
-  tensorflow                1.14.0
-  pytorch                   1.2.0

## Runing
run with 'python main.py --config_filename=data/model/GMSDR_LA.yaml'


##Citing
If you find this resource helpful, please consider to cite our research:

```
@article{zhao2024mtsf-dg,
  title         = {Multiple Time Series Forecasting with Dynamic Graph Modeling},
  author        = {Kai Zhao and Chenjuan Guo and Yunyao Cheng and Peng Han and Miao Zhang and Bin Yang},
  journal       = {Proc. {VLDB} Endow.},
  year          = {2024}
}
```
