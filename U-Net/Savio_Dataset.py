import os
import pandas as pd
from torch.utils.data import Dataset
import multiprocess as multiprocessing
import json
from worker import load_sample_from_savio
import Binvox
class CustomDataset(Dataset):
    def __init__(self, data_path, label_file_path, transform = None, ram_limit = 1000):
        self.data_path = data_path
        self.label_file_path = label_file_path
        self.transform = transform
        self.ram_limit = ram_limit
        self.labels = self.load_labels()
    
    def load_labels(self):
        j = json.load(open(os.path.join(self.data_path, self.label_file_path)))
        return pd.Series(data = j) / 100
    
    def __len__(self):
        return self.labels.size

    def load_sample_into_ram(self, idx):
        num_cores = multiprocessing.cpu_count()
        samples_in_ram = {}
        # need to load ram_limit # of samples, starting from idx
        print('Loading samples', idx)
        labels_to_load = self.labels[idx : min(idx + self.ram_limit, self.__len__())].index.to_list()
        paths_to_load = [os.path.join(self.data_path, label) for label in labels_to_load]
        pool = multiprocessing.Pool(processes = num_cores)
        tuples = pool.map(load_sample_from_savio, paths_to_load)
        pool.close()
        pool.join()
        for sample in tuples:
            samples_in_ram[sample[0]] = sample[1]
        return samples_in_ram

    def load_sample_from_ram(self, file_path):
        return self.samples_in_ram[file_path]
    
    def __getitem__(self, idx):
        # Multi core
        '''
        if idx % self.ram_limit == 0:
            # clear existing samples in ram
            self.samples_in_ram = None
            # need to load new samples into ram
            self.samples_in_ram = self.load_sample_into_ram(idx)
        sample = self.load_sample_from_ram(os.path.join(self.data_path, self.labels.index[idx]))
        if self.transform:
            sample = self.transform(sample)
        if idx % 1000 == 0:
            print('Processing sample number', idx)
        return sample, self.labels.iloc[idx]
        '''
        # Single core
        binvox_name = self.labels.index[idx].replace('rotated_files', 'Binvox_files_default_res')[:-4] + '.binvox'
        sample_path = os.path.join(self.data_path, binvox_name)
        sample = Binvox.read_as_3d_array(open(sample_path, 'rb')).data
        if self.transform:
            sample = self.transform(sample)
        if idx % 1000 == 0:
            print('Processing sample number', idx)
        return sample, self.labels.iloc[idx]