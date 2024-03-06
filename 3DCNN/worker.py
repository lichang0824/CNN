import Binvox
import os
def load_sample_from_disk(file_path):
    return (file_path, Binvox.read_as_3d_array(open(file_path, 'rb')).data)

def load_sample_from_savio(file_path):
    binvox_name = file_path.replace('rotated_files', 'Binvox_files_default_res')[:-4] + '.binvox'
    sample = Binvox.read_as_3d_array(open(binvox_name, 'rb')).data
    return (file_path, sample)