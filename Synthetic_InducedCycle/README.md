# Induced Cycles Synthetic Experiment

This is a python implementation of synthetic experiment on induced cycles, where we compared GKAT with graph attention networks (GAT), graph convolutional networks (GCN)  and Chebyshev spectral graph convolution (SGC). We further compared deeper networks with a two-layer GKAT with different random walk lengths.


## Requirements
This experiment depends on pytorch, as well as some other commonly used packages including numpy, networkx, etc.


## Running the code
 `graph_data`: this folder contains raw train/val networkx graph data and labels, as well as some already calculated frequency matrix and GKAT maksing generated with different random walk lengths

`InducedCycle_GKAT_2layer.ipynb`: this notebook file implements GKAT. 
 
 `Generate_GKAT_masking_and_walks.ipynb`: this notebook generates random walks from train/val graphs under graph_data directory. (There is no need to run this code again, could run `InducedCycle_GKAT_2layer.ipynb` directly with data under graph_data directory already generated from this file)
