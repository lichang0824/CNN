# 3DCNN for UC Berkeley Capstone Project
This is the repository for capstone project #110 (Machine Learning-Based 3D Printing Part Redesign Tool) of the 2023-2024 Master of Engineering program at UC Berkeley. 

This project aims to predict FEM (Finite Element Analysis) results from 3D CAD parts with CNNs using their geometries, specifically CAD parts that are to be 3D printed using the Laser Powder Bed Fusion process. Normally, FEM takes hours or even days to run, and we hope to replace this process with CNN by using it to directly predict the results with good accuracy. Currently, due to the lack of training data, we have made the compromise to predict a single number called unprintability score that simply describes the probability of failure of 3D printing. 

## Code structure

Main CNN architectures and training logics are in the 3DCNN folder, and the scripts to run them are in the root folder. 

Specifically:

3DCNN definitions: [Networks.py](./3DCNN/Networks.py)

Jupyter notebook and python file for training logic: [Savio_3D_CNN.ipynb](./3DCNN/Savio_3D_CNN.ipynb) and [Savio_3D_CNN.py](./3DCNN/Savio_3D_CNN.py). Jupyter Notebook was used to simplify code writing.

Custom dataset for the Binvox files: [Savio_Dataset.py](./3DCNN/Savio_Dataset.py)

Script to generate sbatch scripts for each hyperparameter combination: [make_savio_sh.py](./make_savio_sh.py)

Script to run all scripts generated: [run_savio.py](./run_savio.py)

Configuration file containing the data paths, log file paths, and resolution selection: [Savio_config.json](./Savio_config.json)

Functions to read Binvox voxel files: [Binvox.py](./3DCNN/Binvox.py)

Functions for manipulating filenames: [transform_name.py](./3DCNN/transform_name.py)

## Environment

Dependencies are listed in [environment.yml](./environment.yml). 

## Logging

This project uses `wandb` to log hyperparameter settings, training and validation times, and various CNN performance metrics. An `wandb` account is required and I highly recommend this tool. 

## How to run

1. Create conda environment with the environment file using the `-f` flag: `conda env create -f environment.yml`. Specify environment path with the `-p` flag if necessary.
2. Check paths and settings in [Savio_config.json](./Savio_config.json). 
    1. `data_path` refers to the root folder of training data. 
    2. `train_parts` and `val_parts` refer to  the json files containing the filenames and corresponding unprintability score labels for that file. 
    3. `resolution` denotes the resolution of training data used. Data files in `data_path` must match this resolution. Pay attention to functions in [transform_name.py](./3DCNN/transform_name.py) as they need to transform the raw file names in the json label files into file paths and Binvox file names in the data folder. 
    4. `wandb_path` refers to the path for wandb logging. Consider storing this somewhere else because it can take up significant amount of storage after many runs.
3. Edit [make_savio_sh.py](./make_savio_sh.py) for hyperparameter sweeping. This python script combines sbatch scripts defined in `Preface` with the actual sbatch command that's dynamically generated according to hyperparameter settings, and writes a file containing the finished sbatch script for each hyperparameter combination. Change hyperparameters to sweep in the loops if needed. 
4. Run the script to generate sbatch scripts with `python3 make_savio_sh.py`.
5. Run the generated sbatch scripts with `python3 run_savio.py`.

## TODOs

Add reference to data source
Add reference to Ethan's code for Savio scripting
