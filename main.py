# This code allows to automate the training of all the models needed to reproduce the figure 4
# It allows to do multiple runs. 1 run = 1 figure 4 of the paper without average. The purpose of this is to be able to do
# an average between all the runs and obtain an averaged figure 4s

import os
import numpy as np
from train_helper import execute_training
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__))) # change working directory to the directory where the file is executed

nb_runs = 5 # number of times to repeat the whole procedure to train models for all number of parameters. 

N_param = np.load('parameters_arrays/N_param.npy') # array containing the number of paramters of all models to train within one run
H = np.load('parameters_arrays/H.npy')  # this array contains the hidden layer size for each models to train within one run

##### training parameters #####
epoch_limit = 25
###############################

print('-' * 100)

for run in range(nb_runs):

    start_time = datetime.now() # we compute the time of one run

    for i in range(H.shape[0]):
        print(f'run {run+1}/{nb_runs}, model {i+1}/{H.shape[0]}')
        print(f'target number of parameters neural network: {N_param[i]}')
        print(f'size hidden layer: {H[i]}')
        execute_training(run+1, N_param, H, i, epoch_limit)

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        print(f"time elapsed since start of run{run+1}: {elapsed_time}")
        print('-' * 100)
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"execution time run{run+1}: {elapsed_time}")

    print('-' * 100)

print('done!')







