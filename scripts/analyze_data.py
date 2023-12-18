import numpy as np
import os


def print_npz_info(file_path):
    # TODO: 对数据格式做一个详细的解读
    """ 打印npz文件中数组的信息 """
    data = np.load(file_path)
    print(file_path)
    for arr in data.files:
        print(f"{arr}: shape = {data[arr].shape}, dtype = {data[arr].dtype}")
        print(data[arr][0])



current_dir = os.path.dirname(__file__)
data_file_path1 = os.path.join(current_dir, '..', 'original_labels_Study1', '1_1_1.npz')
data_file_path2 = os.path.join(current_dir, '..', 'original_times_Study1', '1_1_1.npz')
data_file_path3 = os.path.join(current_dir, '..', 'Study1_medfilt11_EMG', '1_1_1_F.npz')
data_file_path4 = os.path.join(current_dir, '..', 'Study1_medfilt11_EMG', '1_1_1_NF.npz')


# 查看 original_labels_Study1 数据
print_npz_info(data_file_path1)

# 查看 original_times_Study1 数据
print_npz_info(data_file_path2)

# 查看 Study1_medfilt11_EMG 数据
print_npz_info(data_file_path3)
print_npz_info(data_file_path4)
