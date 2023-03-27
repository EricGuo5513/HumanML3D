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

import os
import numpy as np
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.tools.omni_tools import copy2cpu as c2c

import shutil, sys
from torch.utils.data import Dataset
import glob
from datetime import datetime
import torch
from human_body_prior.tools.rotation_tools import noisy_zrot
import os.path as osp
import numpy as np
from human_body_prior.tools.omni_tools import logger_sequencer
import pickle
from configer import Configer
from tqdm import tqdm
# import tables as pytables


def dataset_exists(dataset_dir, split_names=None):
    '''
    This function checks whether a valid SuperCap dataset directory exists at a location
    Parameters
    ----------
    dataset_dir

    Returns
    -------

    '''
    if dataset_dir is None: return False
    if split_names is None:
        split_names = ['train', 'vald', 'test']
    import os

    import numpy as np

    done = []
    for split_name in split_names:
        for k in ['root_orient', 'pose_body']:#, 'betas', 'trans', 'joints']:
            outfname = os.path.join(dataset_dir, split_name, '%s.pt' % k)
            done.append(os.path.exists(outfname))
    return np.all(done)

def prepare_vposer_datasets(vposer_dataset_dir, amass_splits, amass_dir, logger=None):

    if dataset_exists(vposer_dataset_dir):
        if logger is not None: logger('VPoser dataset already exists at {}'.format(vposer_dataset_dir))
        return

    ds_logger = log2file(makepath(vposer_dataset_dir, 'dataset.log', isfile=True), write2file_only=True)
    logger = ds_logger if logger is None else logger_sequencer([ds_logger, logger])

    logger('Creating pytorch dataset at %s' % vposer_dataset_dir)
    logger('Using AMASS body parameters from {}'.format(amass_dir))

    shutil.copy2(__file__, vposer_dataset_dir)

    # class AMASS_ROW(pytables.IsDescription):
    #
    #     # gender = pytables.Int16Col(1)  # 1-character String
    #     root_orient = pytables.Float32Col(3)  # float  (single-precision)
    #     pose_body = pytables.Float32Col(21 * 3)  # float  (single-precision)
    #     # pose_hand = pytables.Float32Col(2 * 15 * 3)  # float  (single-precision)
    #
    #     # betas = pytables.Float32Col(16)  # float  (single-precision)
    #     # trans = pytables.Float32Col(3)  # float  (single-precision)

    def fetch_from_amass(ds_names):
        keep_rate = 0.3

        npz_fnames = []
        for ds_name in ds_names:
            mosh_stageII_fnames = glob.glob(osp.join(amass_dir, ds_name, '*/*_poses.npz'))
            npz_fnames.extend(mosh_stageII_fnames)
            logger('Found {} sequences from {}.'.format(len(mosh_stageII_fnames), ds_name))

            for npz_fname in npz_fnames:
                cdata = np.load(npz_fname)
                N = len(cdata['poses'])

                # skip first and last frames to avoid initial standard poses, e.g. T pose
                cdata_ids = np.random.choice(list(range(int(0.1 * N), int(0.9 * N), 1)), int(keep_rate * 0.8 * N), replace=False)
                if len(cdata_ids) < 1: continue
                fullpose = cdata['poses'][cdata_ids].astype(np.float32)
                yield {'pose_body': fullpose[:,3:66], 'root_orient': fullpose[:,:3]}

    for split_name, ds_names in amass_splits.items():
        if dataset_exists(vposer_dataset_dir, split_names=[split_name]): continue
        logger('Preparing VPoser data for split {}'.format(split_name))

        data_fields = {}
        for data in fetch_from_amass(ds_names):
            for k in data.keys():
                if k not in data_fields: data_fields[k] = []
                data_fields[k].append(data[k])

        for k, v in data_fields.items():
            outpath = makepath(vposer_dataset_dir, split_name, '{}.pt'.format(k), isfile=True)
            v = np.concatenate(v)
            torch.save(torch.tensor(v), outpath)

        logger('{} datapoints dumped for split {}. ds_meta_pklpath: {}'.format(len(v), split_name, osp.join(vposer_dataset_dir, split_name)))

    Configer(**{
        'amass_splits':amass_splits.toDict(),
        'amass_dir': amass_dir,
    }).dump_settings(makepath(vposer_dataset_dir, 'settings.ini', isfile=True))

    logger('Dumped final pytorch dataset at %s' % vposer_dataset_dir)