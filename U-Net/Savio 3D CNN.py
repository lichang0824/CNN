#!/usr/bin/env python
# coding: utf-8

# # Preparation

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from Savio_Dataset import CustomDataset
from Networks import ConvNetScalarLabel, count_parameters


# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[ ]:


import wandb
wandb.login()


# # Create dataset

# In[ ]:


def transform(voxel):
    # return torch.unsqueeze(torch.tensor(condense_voxel_array(voxel, 64), dtype = torch.float32), 0)
    return torch.unsqueeze(torch.tensor(voxel, dtype = torch.float32), 0)


# In[ ]:


import json
configs = json.load(open('Savio_config.json', 'r'))
data_path = configs['data_path']
train_parts = configs['train_parts']
val_parts = configs['val_parts']


# In[ ]:


dataset = CustomDataset(data_path = data_path, label_file_path = train_parts, transform = transform)
dataset_val = CustomDataset(data_path = data_path, label_file_path = val_parts, transform = transform)


# In[ ]:


len(dataset)


# # Define Training Logic

# In[ ]:


def train_epoch(model, training_loader, optimizer, loss_fn):
    cumulative_loss = 0.0
    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels)

        # Zero the gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(inputs)

        # Compute loss and its gradients
        # print('label shape', labels.shape)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        cumulative_loss += loss.item()
        
        wandb.log({'batch loss': loss.item()})
    return cumulative_loss / len(training_loader), cumulative_loss


# In[ ]:


def train(config, loss_fn):
    # initialize a wandb run
    wandb.init(config = config, project = 'PAPER', name = '3D CNN Testing')

    # copy the config
    config = wandb.config
    
    print('config:', config)

    # get training loader
    training_loader = DataLoader(dataset, batch_size = config.batch_size, shuffle = False)

    # initialize model
    if config.activation_fn == 'ReLU':
        activation_fn = nn.ReLU()
    
    if config.activation_fn == 'Sigmoid':
        activation_fn = nn.Sigmoid()
    
    model = ConvNetScalarLabel(kernel_size = config.kernel_size, activation_fn = activation_fn).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = config.learning_rate, momentum = 0.9)

    for epoch in range(config.epochs_choice):
        avg_loss_per_batch, cumulative_loss = train_epoch(model, training_loader, optimizer, loss_fn)
        wandb.log({'avg_loss_per_batch': avg_loss_per_batch, 'cumulative_loss': cumulative_loss})
        print(f'Loss for epoch {epoch}: {cumulative_loss}')
    
    return model


# In[ ]:


def test(config, model, loss_fn):
    # copy the config
    config = wandb.config
    
    # get testing loader
    testing_loader = DataLoader(dataset_val, batch_size = config.batch_size, shuffle = False)
    
    testing_loss = 0.0
    y_true = []
    y_pred = []
    for i, data in enumerate(testing_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        testing_loss += loss.item()

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(outputs.cpu().detach().numpy().tolist())
    return testing_loss / len(testing_loader), testing_loss, r2_score(y_true = y_true, y_pred = y_pred)


# In[ ]:


def evaluate(config = None):
    loss_fn = nn.L1Loss()
    model = train(config, loss_fn)
    avg_loss_per_batch_test, testing_loss, r2 = test(config, model, loss_fn)
    wandb.log({'avg_loss_per_batch_test': avg_loss_per_batch_test, 'testing_loss': testing_loss, 'r2': r2})


# # Training settings

# In[ ]:


sweep_config = {
    'method': 'grid'
    }
metric = {
    'name': 'testing_loss',
    'goal': 'minimize'
    }
sweep_config['metric'] = metric
parameters_dict = {
    'kernel_size': {
        'values': [3, 4, 5]
    },
    'activation_fn': {
        'values': ['ReLU', 'Sigmoid']
    },
    'epochs_choice': {
          'values': [5, 10, 20]
    },
    'learning_rate': {
        'values': [1e-4, 1e-3, 1e-2]
    },
    'batch_size': {
        'values': [4]
    },
}

parameters_dict = {
    'kernel_size': {
        'values': [3]
    },
    'activation_fn': {
        'values': ['ReLU']
    },
    'epochs_choice': {
          'values': [20]
    },
    'learning_rate': {
        'values': [1e-4]
    },
    'batch_size': {
        'values': [4]
    },
}

sweep_config['parameters'] = parameters_dict


# # Start

# In[ ]:


sweep_id = wandb.sweep(sweep_config, project = 'CNN_sweep_scalar')


# In[ ]:


wandb.agent(sweep_id = sweep_id, function = evaluate)


# In[ ]:




