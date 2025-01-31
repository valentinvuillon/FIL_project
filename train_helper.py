# This code contains the logic to train one model within one run

from helpers import ImageDataset, FCNeuralNetwork, expand_network_weights, train_model
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import os


def execute_training(run, N_param, H, i, epoch_limit):
    path = f'runs/run{run}/H{H[i]}'
    Path(path).mkdir(parents=True, exist_ok=True)

    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    x_train_subsampled=torch.load('dataset/x_train_subsampled.pt')
    y_train_subsampled=torch.load('dataset/y_train_subsampled.pt')

    x_test_subsampled=torch.load('dataset/x_test_subsampled.pt')
    y_test_subsampled=torch.load('dataset/y_test_subsampled.pt')

    K = 10 # Number of classes

    train_dataset = ImageDataset(x_train_subsampled, y_train_subsampled, K)
    test_dataset = ImageDataset(x_test_subsampled, y_test_subsampled, K)

    train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = False)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 512, shuffle = False)


    if(i==0):
        print(f'case 0') # sanity check to know that the right case has been chosen
        over_parametrized = False
        model = FCNeuralNetwork(H=H[i], K=K, Glorot_initialization = True).to(device)

        print(f'number of parameters model: {model.count_parameters()}') #sanity check
        train_model(model, train_loader, test_loader, device, over_parametrized, epoch_limit, path)

    else:
        if(N_param[i] < K*len(train_dataset)):
            print(f'case 1') # sanity check to know that the right case has been chosen
            over_parametrized = False
            smaller_model = FCNeuralNetwork(H=H[i-1], K=K, Glorot_initialization = False).to(device)
            smaller_model.load_state_dict(torch.load(f'runs/run{run}/H{H[i-1]}/model.pt'))
            model = FCNeuralNetwork(H=H[i], K=K, Glorot_initialization = False).to(device)
            expand_network_weights(smaller_model, model, H[i-1], H[i])

            print(f'smaller model number of parameters: {smaller_model.count_parameters()}') #sanity check
            print(f'larger model number of parameters: {model.count_parameters()}') #sanity check
            train_model(model, train_loader, test_loader, device, over_parametrized, epoch_limit, path)
            
        else:
            print(f'case 2') # sanity check to know that the right case has been chosen
            over_parametrized = True
            model = FCNeuralNetwork(H=H[i], K=K, Glorot_initialization = False).to(device)

            print(f'number of parameters model: {model.count_parameters()}') #sanity check
            train_model(model, train_loader, test_loader, device, over_parametrized, epoch_limit, path)



