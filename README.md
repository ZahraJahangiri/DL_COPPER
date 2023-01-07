# DL_COPPER
PyTorch implementation for the paper "A Deep Learning approach for exploring the design space for the decarbonization of the Canadian electricity system"
# Introduction
A supervised machine learning surrogate of the Canadian electricity system design space 76
using a surrogate of the Canadian Opportunities for Planning and Production of Electricity Resources (COPPER) model is developed based on residual neural networks, accurately approximating the model’s outputs while reducing the computation cost by five orders of magnitude. This increased efficiency enables the evaluation of the outputs' sensitivity to the inputs, allowing the evaluation of system development factors relationships for the Canadian electricity system between 2030 and 2050.
# Requirements
* Python3.
* [PyTorchLightening](https://www.pytorchlightning.ai/)
# Data Generation
A dataset was generated of COPPER simulations to train, validate, and test surrogate NN options. We used COPPER V5 to generate a dataset containing 1000 COPPER runs.
1000 random samples using [sample generator](https://github.com/ZahraJahangiri/DL_COPPER/blob/main/DataGen/input_sample_generator_v1.1.py) are generated. We ran [COPPER5.1.py](https://github.com/ZahraJahangiri/DL_COPPER/blob/main/DataGen/COPPER5.1.py) for these inputs. 
# Data Manipulation and Model Selection 
The process of data cleanup and model development and evaluation is provided [here](https://github.com/ZahraJahangiri/DL_COPPER/tree/main/DataPrep_ModelDev). For model development, we used gpu provided by Compute Canada. 
# Clustering the results 
we used K-Means clustering, t-SNE dimensionality reduction to be able to visualize 2000 plots, resulting from running the developed model. The codes for clustering the results and visualizing them using heatmaps are provided [here](https://github.com/ZahraJahangiri/DL_COPPER/tree/main/Visualization).
