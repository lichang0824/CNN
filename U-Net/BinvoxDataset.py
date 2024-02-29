import os
import Binvox
import pandas as pd
import math
from torch.utils.data import Dataset
import multiprocess as multiprocessing
import json
class CustomDataset(Dataset):
    def __init__(self, input_folder_path, input_folder_name, label_file_path, transform = None, max_count = None, ram_limit = 1000, label_type = 'json'):
        self.input_folder_path = input_folder_path
        self.input_folder_name = input_folder_name
        self.label_file_path = label_file_path
        self.transform = transform
        self.max_count = max_count if max_count else math.inf
        self.ram_limit = ram_limit
        self.label_type = label_type
        self.labels = self.load_labels()
        self.input_paths, self.input_names = self.load_input_paths_names()
    
    def load_input_paths_names(self):
        count = 0
        list_of_file_paths = []
        list_of_file_names = []
        directory = os.fsencode(self.input_folder_path)
        for file in os.listdir(directory):
            if (self.label_type == 'json') and ('data/' + self.input_folder_name + '/rotated_files/' + os.fsdecode(file).replace('binvox', 'stl') not in self.labels.index):
                continue
            list_of_file_paths.append(self.input_folder_path + os.fsdecode(file))
            list_of_file_names.append(os.fsdecode(file))
            count += 1
            if count >= self.max_count:
                break
        return list_of_file_paths, list_of_file_names
    
    def load_labels(self):
        if self.label_type == 'csv':
            csv = pd.read_csv(self.label_file_path, header = None)
            return pd.Series(data = csv[1].values, index = csv[0].values, dtype = 'float32')
        if self.label_type == 'json':
            j = json.load(open(self.label_file_path))
            return pd.Series(data = j)
    
    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):
        if idx % self.ram_limit == 0:
            # clear existing samples in ram
            self.samples_in_ram = None
            # need to load new samples into ram
            self.samples_in_ram = self.load_sample_into_ram(idx)
        sample = self.load_sample_from_ram(self.input_names[idx])
        
        # debug
        # print(self.input_names[idx].replace('binvox', 'stl'))
        
        if self.transform:
            sample = self.transform(sample)
        if idx % self.ram_limit == 0:
            print('Processing sample number', idx)
        if self.label_type == 'csv':
            return sample, self.labels[self.input_names[idx].replace('binvox', 'stl')]
        if self.label_type == 'json':
            return sample, float(self.labels['data/' + self.input_folder_name + '/rotated_files/' + self.input_names[idx].replace('binvox', 'stl')])
    
    def load_sample_from_disk(self, file_name):
        return (file_name, Binvox.read_as_3d_array(open(self.input_folder_path + file_name, 'rb')).data)

    def load_sample_into_ram(self, idx):
        num_cores = multiprocessing.cpu_count()
        samples_in_ram = {}
        # need to load ram_limit # of samples, starting from idx
        print('Loading samples', idx, 'through', min(idx + self.ram_limit, self.__len__()) - 1)
        sample_names_to_load = self.input_names[idx : min(idx + self.ram_limit, self.__len__())]

        pool = multiprocessing.Pool(processes = num_cores)
        tuples = pool.map(self.load_sample_from_disk, sample_names_to_load)
        '''
        results = [pool.apply_async(self.load_sample_from_disk, args = (sample_name,)) for sample_name in sample_names_to_load]
        tuples = [result.get() for result in results]
        '''
        pool.close()
        pool.join()
        '''
        for file_name in self.input_names[idx : min(idx + self.ram_limit, self.__len__())]:
            samples_in_ram[file_name] = self.load_sample_from_disk(file_name)
        '''
        for sample in tuples:
            samples_in_ram[sample[0]] = sample[1]
        return samples_in_ram

    def load_sample_from_ram(self, file_name):
        return self.samples_in_ram[file_name]
