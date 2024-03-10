import re
def get_binvox_name_256(file_name):
    return file_name.replace('rotated_files', 'Binvox_files_default_res')[:-3] + 'binvox'

def get_binvox_name_64(file_name):
    file_name = file_name.replace('rotated_files', 'Binvox_files_64_res_compressed')
    name = re.search('.{8}-.{4}-.{4}-.{4}-.{12}', file_name)[0]
    file_name = file_name.replace(name, name + '_compressed')
    file_name = file_name[:-3] + 'binvox'
    return file_name