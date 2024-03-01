import Binvox
def load_sample_from_disk(file_path):
    return (file_path, Binvox.read_as_3d_array(open(file_path, 'rb')).data)