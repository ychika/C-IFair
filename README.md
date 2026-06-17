This repository contains the source code for the following paper:

> Yoichi Chikahara. "Fairness under Graph Uncertainty: Achieving Interventional Fairness with Partially Known Causal Graphs over Clusters of Variables", The 42nd Conference on Uncertainty in Artificial Intelligence (UAI), 2026. [[paper]](https://arxiv.org/abs/2602.23611)



### Install libraries

1. Use conda to create python environment (Python=3.8):

$ conda env create -n proj-C-IFair -f environment.yml 

2. Activate the environment:

$ conda activate proj-C-IFair

3. Use pip to install the required python packages:

$ python -m pip install -r requirements.txt 

## Run experiments

$ ./run.sh

## Settings

The main settings are defined in `run.sh`. You can modify the following settings as needed. The default settings are for linear synthetic datasets.

DATANAME=linear
#DATANAME=nonlinear
#DATANAME=adult
#DATANAME=german
#DATANAME=lin_conn
SEED=24
N=5000
EPOCHS=1000
PROP_STEPS=500
LMBDA=5.0 
D_CLUSTERS=5

## License 

This source code is licensed under MIT license. 


