# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.01.02

import glob, os

import torch
from torch.utils.data import Dataset
from configer import Configer

class VPoserDS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir, data_fields=[]):
        assert os.path.exists(dataset_dir)
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            if len(data_fields) != 0 and k not in data_fields: continue
            self.ds[k] = torch.load(data_fname).type(torch.float32)

        dataset_ps_fname = glob.glob(os.path.join(dataset_dir, '..', '*.ini'))
        if len(dataset_ps_fname):
            self.ps = Configer(default_ps_fname=dataset_ps_fname[0], dataset_dir=dataset_dir)

    def __len__(self):
        k = list(self.ds.keys())[0]
        return len(self.ds[k])

    def __getitem__(self, idx):
        return self.fetch_data(idx)

    def fetch_data(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        return data

