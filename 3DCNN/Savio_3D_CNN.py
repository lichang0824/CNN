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
import time
import argparse


# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[ ]:


import wandb
wandb.login()


# # Calculate number of parameters

# In[ ]:


count_parameters(ConvNetScalarLabel())


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


# In[ ]:


len(dataset_val)


# # Define Training Logic

# In[ ]:


def train_epoch(model, training_loader, loss_fn, optimizer):
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
        loss = loss_fn(outputs, labels.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Increment by the mean loss of the batch
        cumulative_loss += loss.item()
        
        wandb.log({'batch loss': loss.item()})
    return cumulative_loss / len(training_loader)


# In[ ]:


def validate(model, validation_loader, loss_fn):
    validation_loss = 0.0
    y_true = []
    y_pred = []
    for i, data in enumerate(validation_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels.float())
        validation_loss += loss.item()

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(outputs.cpu().detach().numpy().tolist())
    return validation_loss / len(validation_loader), r2_score(y_true = y_true, y_pred = y_pred)


# In[ ]:


def evaluate(args, loss_fn):
    
    # get training loader
    training_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False)

    # get validation loader
    validation_loader = DataLoader(dataset_val, batch_size = args.batch_size, shuffle = False)

    # initialize model
    if args.activation_fn == 'ReLU':
        activation_fn = nn.ReLU()
    
    if args.activation_fn == 'Sigmoid':
        activation_fn = nn.Sigmoid()
    
    model = ConvNetScalarLabel(kernel_size = args.kernel_size, activation_fn = activation_fn).to(device)
    print(count_parameters(model))
    
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9)

    for epoch in range(args.epochs_choice):
        wandb.log({'epoch': epoch})
        # train
        tic = time.time()
        train_loss = train_epoch(model, training_loader, loss_fn, optimizer)
        toc = time.time()
        wandb.log({'train_loss': train_loss, 'train_time': round(toc - tic)})
        print(f'Train loss for epoch {epoch}: {train_loss}, cumulative loss for epoch {epoch}: {cumulative_loss * args.batch_size}, train time for epoch {epoch}: {round(toc - tic)}')

        # validate
        tic = time.time()
        validation_loss, r2 = validate(model, validation_loader, loss_fn)
        toc = time.time()
        wandb.log({'validation_loss': validation_loss, 'r2': r2, 'validate_time': round(toc - tic)})
        print(f'Validate loss for epoch {epoch}: {validation_loss}, r2 for epoch {epoch}: {r2}, validate time for epoch {epoch}: {round(toc - tic)}')
    
    return model


# In[ ]:


def run(args = None):
    training_set = train_parts[5:-5]
    name = f'3DCNN_{training_set}_{args.kernel_size}_{args.activation_fn}_{args.epochs_choice}_{args.learning_rate}_{args.batch_size}'
    print(name)

    config = {
        # filename of the training set, '10000' in 'data/10000.json' for example
        'training_set': training_set,
        'kernel_size': args.kernel_size,
        'activation_fn': args.activation_fn,
        'epochs_choice': args.epochs_choice,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'architecture': '3DCNN_256_stride1'
    }
    # initialize a wandb run
    wandb.init(name = name, project = 'PAPER', config = config)
    
    loss_fn = nn.L1Loss(reduction = 'mean')
    model = evaluate(args, loss_fn)
    torch.save(model, 'model.pt')


# # Parsing parameters

# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('--kernel_size', type = int, required = True)
parser.add_argument('--activation_fn', type = str, required = True)
parser.add_argument('--epochs_choice', type = int, required = True)
parser.add_argument('--learning_rate', type = float, required = True)
parser.add_argument('--batch_size', type = int, required = True)
args = parser.parse_args()


# # Start

# In[ ]:


run(args = args)
wandb.finish()

