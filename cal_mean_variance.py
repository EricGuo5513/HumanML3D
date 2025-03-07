"""
Calculate the mean and variance of sequence vectors for a given rig type.

# each frame is described in four broad sections given a rig with j joints:
# 0 -- 3                                : global position, 4
# 4 -- 4 + (j - 1) * 3                  : local positions of other joints with respect to root, (j - 1) * 3
# 4 + (j - 1) * 3 -- 4 + (j - 1) * 9    : local rotations of other joints with respect to root, (j - 1) * 6
# 4 + (j - 1) * 9 -- end                : global joint velocities plus foot contact information, j * 3 + 4
# see https://arxiv.org/pdf/2405.11126 Appendix A for more details
# from the computed data, this corresponds to:
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)

"""
import os
from os.path import join as pjoin
import numpy as np


def mean_variance(data_dir: str, save_dir: str, joints_num: int):
    """
    Compute the mean and variance of the joint vector sequences in a given directory.

    :param data_dir:    string path to the vectors whose mean and variance will be computed
    :param save_dir:    string path to the directory where the results will be stored
    :param joints_num:  integer number of joints for a given rig type
    """
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    section_one = 4 + (joints_num - 1) * 3
    section_two = 4 + (joints_num - 1) * 9
    section_three = section_two + joints_num * 3
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: section_one] = Std[4: section_one].mean() / 1.0
    Std[section_one: section_two] = Std[section_one: section_two].mean() / 1.0
    Std[section_two: section_three] = Std[section_two: section_three].mean() / 1.0
    Std[section_three: ] = Std[section_three: ].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean_abs_3d.npy'), Mean)
    np.save(pjoin(save_dir, 'Std_abs_3d.npy'), Std)

    return Mean, Std

if __name__ == '__main__':
    data_dir = './Custom/new_joint_vecs/'
    save_dir = './Custom/'
    mean, std = mean_variance(data_dir=data_dir, save_dir=save_dir, joints_num=27)
