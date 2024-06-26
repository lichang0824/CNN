#!/usr/bin/env python
# coding: utf-8

# Preparation

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from Savio_Dataset import CustomDataset
from Networks import ConvNetScalarLabel256, ConvNetScalarLabel64, MLPBaseline64, count_parameters
import time
import argparse
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import wandb
wandb.login()

# Parsing parameters

parser = argparse.ArgumentParser()
parser.add_argument('--kernel_size', type = int, required = True)
parser.add_argument('--activation_fn', type = str, required = True)
parser.add_argument('--epochs_choice', type = int, required = True)
parser.add_argument('--learning_rate', type = float, required = True)
parser.add_argument('--batch_size', type = int, required = True)
args = parser.parse_args()

# Create dataset

def transform(voxel):
    # return torch.unsqueeze(torch.tensor(condense_voxel_array(voxel, 64), dtype = torch.float32), 0)
    return torch.unsqueeze(torch.tensor(voxel, dtype = torch.float32), 0)

import json
configs = json.load(open('Savio_config.json', 'r'))
data_path = configs['data_path']
train_parts = configs['train_parts']
val_parts = configs['val_parts']
resolution = configs['resolution']
wandb_path = configs['wandb_path']
baseline = configs['baseline']

dataset = CustomDataset(data_path = data_path, label_file_path = train_parts, transform = transform, resolution = resolution)
dataset_val = CustomDataset(data_path = data_path, label_file_path = val_parts, transform = transform, resolution = resolution)

len(dataset)

len(dataset_val)

# Get Model Class

if baseline == 0:
    if resolution == 256:
        model_class = ConvNetScalarLabel256
    if resolution == 64:
        model_class = ConvNetScalarLabel64
if baseline == 1:
    if resolution == 64:
        model_class = MLPBaseline64

# Define Training Logic

def train_epoch(model, training_loader, loss_fn, optimizer):
    model.train()
    cumulative_loss = 0.0
    cumulative_time = 0.0
    cumulative_load_time = 0.0
    load_start = time.time()
    for i, data in enumerate(tqdm(training_loader)):
        load_end = time.time()
        cumulative_load_time += load_end - load_start
        wandb.log({'load_time_train_ms': (load_end - load_start) * 1000})
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels)

        tic = time.time()
        
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

        toc = time.time()

        # Increment by the mean loss of the batch
        cumulative_loss += loss.item()

        if i > 10:
            cumulative_time += toc - tic
            wandb.log({'inference_time_train_ms': (toc - tic) * 1000})
        
        wandb.log({'batch loss': loss.item()})
        load_start = time.time()
    return cumulative_loss / len(training_loader), cumulative_time / (len(training_loader) - 10) * 1000, cumulative_load_time / len(training_loader) * 1000

def validate(model, validation_loader, loss_fn):
    model.eval()
    validation_loss = 0.0
    validation_time = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(validation_loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            tic = time.time()
            outputs = model(inputs)
            toc = time.time()
            loss = loss_fn(outputs, labels.float())
            validation_loss += loss.item()

            if i > 10:
                validation_time += toc - tic
                wandb.log({'inference_time_validation_ms': (toc - tic) * 1000})
    
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(outputs.cpu().detach().numpy().tolist())
    return validation_loss / len(validation_loader), validation_time / (len(validation_loader) - 10) * 1000, r2_score(y_true = y_true, y_pred = y_pred)

def evaluate(args, loss_fn):
    
    # get training loader
    training_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)

    # get validation loader
    validation_loader = DataLoader(dataset_val, batch_size = args.batch_size, shuffle = True)

    # initialize model
    if args.activation_fn == 'ReLU':
        activation_fn = nn.ReLU()
    
    if args.activation_fn == 'Sigmoid':
        activation_fn = nn.Sigmoid()
    
    model = model_class(kernel_size = args.kernel_size, activation_fn = activation_fn).to(device)
    print(count_parameters(model))
    
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = 0.9)

    for epoch in range(args.epochs_choice):
        wandb.log({'epoch': epoch})
        # train
        tic = time.time()
        train_loss, inference_time, load_time = train_epoch(model, training_loader, loss_fn, optimizer)
        toc = time.time()
        wandb.log({'train_loss': train_loss, 'train_time': round(toc - tic), 'train_time_batch_ms': inference_time, 'load_time_batch_ms': load_time})
        print(f'Train loss for epoch {epoch}: {train_loss}, train time for epoch {epoch}: {round(toc - tic)}, average inference time for batch in milliseconds: {inference_time}')

        # validate
        tic = time.time()
        validation_loss, inference_time, r2 = validate(model, validation_loader, loss_fn)
        toc = time.time()
        wandb.log({'validation_loss': validation_loss, 'r2': r2, 'validate_time': round(toc - tic), 'validate_time_batch_ms': inference_time})
        print(f'Validate loss for epoch {epoch}: {validation_loss}, r2 for epoch {epoch}: {r2}, validate time for epoch {epoch}: {round(toc - tic)}, average inference time for batch in milliseconds: {inference_time}')
    
    return model

def run(args = None):
    training_set = train_parts[5:-5]
    name = f'3DCNN_{training_set}_{args.kernel_size}_{args.activation_fn}_{args.epochs_choice}_{args.learning_rate}_{args.batch_size}'
    # name = '3DCNN runtime testing'
    print(name)

    config = {
        # filename of the training set, '10000' in 'data/10000.json' for example
        'training_set': training_set,
        'kernel_size': args.kernel_size,
        'activation_fn': args.activation_fn,
        'epochs_choice': args.epochs_choice,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'architecture': model_class.arch,
        'num_parameters': count_parameters(model_class(kernel_size = args.kernel_size))
    }
    # initialize a wandb run
    wandb.init(name = name, project = 'PAPER', config = config, dir = wandb_path)
    
    loss_fn = nn.L1Loss(reduction = 'mean')
    model = evaluate(args, loss_fn)
    torch.save(model, 'model.pt')

# # Start

run(args = args)
wandb.finish()

