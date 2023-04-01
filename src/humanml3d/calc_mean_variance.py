import os
from os.path import join as pjoin

import numpy as np

from tqdm import tqdm


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, joints_num):
    file_list = os.listdir(data_dir)

    data_list = []
    for file in tqdm(file_list):
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue

        data_list.append(data)

    n = 0
    mean = 0
    for data in data_list:
        for i in range(data.shape[0]):
            mean += data[i]
            n += 1
    mean /= n

    variance = 0
    for data in data_list:
        for i in range(data.shape[0]):
            variance += (data[i] - mean) ** 2
    std = np.sqrt(variance / n)

    std[0:1] = std[0:1].mean() / 1.0
    std[1:3] = std[1:3].mean() / 1.0
    std[3:4] = std[3:4].mean() / 1.0
    std[4 : 4 + (joints_num - 1) * 3] = std[4 : 4 + (joints_num - 1) * 3].mean() / 1.0
    std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] = (
        std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9].mean() / 1.0
    )
    std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3] = (
        std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3].mean()
        / 1.0
    )
    std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = (
        std[4 + (joints_num - 1) * 9 + joints_num * 3 :].mean() / 1.0
    )

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == std.shape[-1]

    return mean, std
