import os
import Binvox
import pandas as pd
from torch.utils.data import Dataset
import json
class CustomDataset(Dataset):
    def __init__(self, data_path, label_file_path, transform = None):
        self.data_path = data_path
        self.label_file_path = label_file_path
        self.transform = transform
        self.labels = self.load_labels()
        # self.input_paths, self.input_names = self.load_input_paths_names()

    """
    def load_input_paths_names(self):
        count = 0
        list_of_file_paths = []
        list_of_file_names = []
        directory = os.fsencode(self.input_folder_path)
        for file in os.listdir(directory):
            if (self.label_type == 'json') and ('data/parts_0, files 1 through 3950/rotated_files/' + os.fsdecode(file).replace('binvox', 'stl') not in self.labels.index):
                continue
            list_of_file_paths.append(self.input_folder_path + os.fsdecode(file))
            list_of_file_names.append(os.fsdecode(file))
            count += 1
            if count >= self.max_count:
                break
        return list_of_file_paths, list_of_file_names
    """
    
    def load_labels(self):
        j = json.load(open(os.path.join(self.data_path, self.label_file_path)))
        return pd.Series(data = j)
    
    def __len__(self):
        return self.labels.size
    
    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_path, self.labels.index[idx].replace('.stl', '.binvox'))
        sample = Binvox.read_as_3d_array(open(sample_path, 'rb')).data
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels.iloc[idx]
        # debug
        # print(self.input_names[idx].replace('binvox', 'stl'))
        """
        if self.transform:
            sample = self.transform(sample)
        if idx % self.ram_limit == 0:
            print('Processing sample number', idx)
        if self.label_type == 'csv':
            return sample, self.labels[self.input_names[idx].replace('binvox', 'stl')]
        if self.label_type == 'json':
            return sample, self.labels['data/parts_0, files 1 through 3950/rotated_files/' + self.input_names[idx].replace('binvox', 'stl')]
    
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
"""