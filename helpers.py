# This file contains definitions of classes and function necessary to train a model

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, x, y, K):
        self.x, self.y = x, y
        self.x = self.x / 255.  # We normalize the image amplitude between 0 and 1
        self.y = F.one_hot(self.y, num_classes=K).to(torch.float) # One hot encoding
    def __len__(self): 
        return self.x.shape[0]
    def __getitem__(self, ix): 
        return self.x[ix], self.y[ix]
    

class FCNeuralNetwork(nn.Module):
    def __init__(self, H, K, Glorot_initialization):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, H)
        self.Matrix2 = nn.Linear(H, K)
        self.R = nn.ReLU()

        if(Glorot_initialization == True):
            # Initialize weights using Glorot Uniform
            self._initialize_weights()

    def forward(self,x):
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.Matrix2(x)
        return x.squeeze()
    
    def _initialize_weights(self):
        # Apply Xavier Uniform initialization to weights
        init.xavier_uniform_(self.Matrix1.weight)
        init.xavier_uniform_(self.Matrix2.weight)
        
        # Xavier initialization applies only to weights, not biases, so we set them to 0
        init.zeros_(self.Matrix1.bias)
        init.zeros_(self.Matrix2.bias)

    def count_parameters(self): # method to count the number of parameters of the network
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

def expand_network_weights(smaller_net, larger_net, H1, H2):
    """
    Expand the weights from smaller_net (with H1 hidden units) into larger_net (with H2 hidden units).
    
    Arguments:
    - smaller_net: The trained smaller network (H=H1).
    - larger_net: The larger network to initialize (H=H2).
    - H1: Number of hidden units in the smaller network.
    - H2: Number of hidden units in the larger network (H2 > H1).
    """
    with torch.no_grad():
        #-----Matrix1-----#

        # weights 

        # The first H1 lines of Matrix1.weight of the larger network are a copy of Matrix1.weight of the smaller network
        larger_net.Matrix1.weight[:H1, :] = smaller_net.Matrix1.weight

        # The rest of the lines of Matrix.weight of the larger network are randomly initialized
        nn.init.normal_(larger_net.Matrix1.weight[H1:], mean=0, std=0.01)

        #-----------------#

        # biases

        # The first H1 elements of Matrix1.bias of the larger network are a copy of Matrix1.bias of the smaller network
        larger_net.Matrix1.bias[:H1] = smaller_net.Matrix1.bias

        # The rest of the elements of Matrix1.bias of the larger model are randomly initialized
        nn.init.normal_(larger_net.Matrix1.bias[H1:], mean=0, std=0.01)

        #-----Matrix2-----#

        # weights

        # The first H1 columns of Matrix2.weight of the larger network are a copy of Matrix2.weight of the smaller network
        larger_net.Matrix2.weight[:, :H1] = smaller_net.Matrix2.weight

        # The rest of the columns of Matrix2.weight of the larger network are randomly initialized
        nn.init.normal_(larger_net.Matrix2.weight[:, H1:], mean=0, std=0.01)

        #-----------------#
        
        # biases

        # Since a change from H1 to H2 in the hidden layer doesn't affect the output dim (K=10), the biases of Matrix2 of the larger model can simply be a copy of the bias of the smaller model
        larger_net.Matrix2.bias = smaller_net.Matrix2.bias


class ZeroOneLoss(nn.Module):
    def __init__(self):
        super(ZeroOneLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of predicted class indices (shape: [batch_size]).
            targets: Tensor of ground truth class indices (shape: [batch_size]).
        Returns:
            Zero-One Loss (scalar tensor).
        """
        # Ensure predictions and targets are of the same shape
        assert predictions.shape == targets.shape, "Shape mismatch between predictions and targets"
        
        # Count incorrect predictions
        incorrect = (predictions != targets).float()
        
        # Compute the mean zero-one loss
        loss = incorrect.mean()
        return loss
    

def train_model(model, train_loader, test_loader, device, over_parametrized, epoch_limit, path):

    criterion1 = nn.MSELoss()
    criterion2 = ZeroOneLoss()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01,  momentum=0.95)

    #scheduler
    n_epoch_scheduler = 500 # each n_epoch_scheduler the learning rate is decreased by 10%

    def lr_lambda(epoch):
        return 0.9 ** (epoch // n_epoch_scheduler)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_squared_loss = [] # contains training squared loss for each epoch
    val_squared_loss = [] # contains validation squared loss for each epoch

    train_zero_one_loss = [] # contains training zero-one loss for each epoch
    val_zero_one_loss = [] # contains validation zero-one loss for each epoch

    file_path =  os.path.join(path, 'training_info.txt')
    file = open(file_path, 'a')
    file.truncate(0) #necessary to clear the content of the .txt file

    file.write(f'Training started at {datetime.now().strftime("%H:%M:%S")}\n')

    for epoch in tqdm(range(epoch_limit), desc='Training Progress', ncols=100):
        model.train()
        train_squared_loss_batch = [] # contains training squared loss for each batch for one epoch
        train_zero_one_loss_batch = [] # contains training zero-one loss for each batch for one epoch

        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            squared_loss = criterion1(outputs, y_batch)
            zero_one_loss = criterion2(outputs.argmax(dim = 1), y_batch.argmax(dim = 1))

            train_squared_loss_batch.append(squared_loss.item())
            train_zero_one_loss_batch.append(zero_one_loss.item())

            squared_loss.backward()
            optimizer.step()

        train_squared_loss.append(np.mean(train_squared_loss_batch))
        train_zero_one_loss.append(np.mean(train_zero_one_loss_batch))

        # Validation
        model.eval()

        val_squared_loss_batch = [] # contains validation squared loss for each batch for one epoch
        val_zero_one_loss_batch = [] # contains validation zero-one loss for each batch for one epoch

        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                squared_loss = criterion1(outputs, y_batch)
                zero_one_loss = criterion2(outputs.argmax(dim = 1), y_batch.argmax(dim = 1))

                val_squared_loss_batch.append(squared_loss.item())
                val_zero_one_loss_batch.append(zero_one_loss.item())

        val_squared_loss.append(np.mean(val_squared_loss_batch))
        val_zero_one_loss.append(np.mean(val_zero_one_loss_batch))

        scheduler.step()
        
        if ((over_parametrized == False) and (epoch % n_epoch_scheduler == 0)):
            lr = scheduler.get_last_lr()[0]

        torch.save(model.state_dict(),  os.path.join(path, 'model.pt')) #saving model

        file.write(f'Epoch {epoch + 1}/{epoch_limit}\n')
        file.write(f'Train Squared Loss: {train_squared_loss[epoch]:.7f}, Validation Squared Loss: {val_squared_loss[epoch]:.7f}\n')
        file.write(f'Train Zero One Loss: {train_zero_one_loss[epoch]:.7f}, Validation Zero One Loss: {val_zero_one_loss[epoch]:.7f}\n')
        file.write(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")
        file.flush()  #to write the data to the file without closing the file

        #saving training data
        train_squared_loss_numpy = np.array(train_squared_loss)
        val_squared_loss_numpy = np.array(val_squared_loss)
        train_zero_one_loss_numpy = np.array(train_zero_one_loss)
        val_zero_one_loss_numpy = np.array(val_zero_one_loss)
        
        np.save(os.path.join(path, 'train_squared_loss_numpy.npy'), train_squared_loss_numpy)
        np.save(os.path.join(path, 'val_squared_loss_numpy.npy'), val_squared_loss_numpy)
        np.save(os.path.join(path, 'train_zero_one_loss_numpy.npy'), train_zero_one_loss_numpy)
        np.save(os.path.join(path, 'val_zero_one_loss_numpy.npy'), val_zero_one_loss_numpy)

        if ((over_parametrized == False) and (train_zero_one_loss[epoch] == 0)):
            file.write('Early stopping\n')
            file.flush() 
            print('Early stopping')
            break

    file.write(f'Training ended at {datetime.now().strftime("%H:%M:%S")}\n')

    file.close()