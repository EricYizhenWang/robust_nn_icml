# Robust 1-Neareset Neighbor Algorithm for Binary Classification
This repository contains a python implementation of the robust 1-NN algorithm descriobe in paper "Analyzing the Robustness of Nearest Neighbors to Adversarial Examples" accepted by ICML 2018. The paper can be found on arXiv at https://arxiv.org/abs/1706.03922

## File Decriptions
1. mnist.sh
   This is an example script running the MNIST experiment appeared in the paper and producing corresponding plots/figures.
 
2. run_experiment.py
   This scripts implements the 4 baselines: standard_nn, robust_nn, ATnn and ATnn-all. The script takes two arguments, which are the data set and the attack method.
   Legit data set arguments are: halfmoon, mnist, abalone
   Legit attack method arguments are: wb (direct white-box), wb_kernel (white-box attack on a kernel substitute), kernel (black-box attack on a kernel substitute) and nn (black-box attack against on a neural net substitute.)
   The scripts will run the chosen attack on the data set and save results of the 4 baselines.
   
   Model parameters, size of data set and the number of repeated experiments can be set in the script as well.
   
3. nn_attack_white_box.py
   This module contains attack methods for white-box attacks.
   
4. nn_attacks.py
   This module is an integration of all attack methods.
   
5. eps_separation.py
   This module contains the method finding epsilon-separated subset.
   
6. robust_1nn.py
   This module contains the implementaion of robust_1nn algorithm.
 
7. prepare_data.py
   This module contains method generating data set of desired size.
   
8. model_utils.py
   This module contains the definition of models used in the experiment.
   
9. plotting.py
   This script plots the MNIST result.
   
## Running time
For MNIST, an example run of a single experiment using a total of 1000 training images and 400 test images takes ~1hr on a 7770k+1080ti desktop.

## Required Environment
   1. standard numpy and matlibplot packages
   2. tensorflow with gpu
   3. hopcroftkarp module of finding maximum matching.  (can be added using pip install hopcroftkarp)
   4. cleverhans adversarial attack package. (can be found at https://github.com/tensorflow/cleverhans)
   
   For convenience and for avoiding possible version inconsistency, 3 and 4's source code are also included in this repository. Please follow the original author's instruction and license agreements.
   
## Contact for Further Clarification
   Feel free to contact me at yiw248@eng.ucsd.edu
