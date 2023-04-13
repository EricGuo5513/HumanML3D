import os
import os.path as osp

from pathlib import Path

import numpy as np

import torch
from torch import nn

from einops import rearrange, repeat

from .body_models import BodyModel


class AMASSBodyModel(nn.Module):
    """Body Model Used in AMASS dataset

    Args:
        male_bm_path: path to male smplh body model npz file
        female_bm_path: path to male smplh body model npz file
        male_dmpl_path: path to male dmpl body model npz file
        female_dmpl_path: path to male dmpl body model npz file
        num_betas: number of body paramters, defaults to 10
        num_dmpls: number of DMPL paramters, defaults to 8
    """

    def __init__(
        self,
        male_bm_path: Path,
        female_bm_path: Path,
        male_dmpl_path: Path,
        female_dmpl_path: Path,
        num_betas: int = 10,
        num_dmpls: int = 8,
    ) -> None:
        super().__init__()

        self.male_bm = BodyModel(
            bm_fname=str(male_bm_path),
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=str(male_dmpl_path),
        )

        self.female_bm = BodyModel(
            bm_fname=str(female_bm_path),
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=str(female_dmpl_path),
        )

    def forward(self, trans, gender, mocap_framerate, poses, betas, ex_fps=20):
        fps = mocap_framerate
        frame_number = trans
        if fps == 0 or frame_number is None:
            return 0
        else:
            frame_number = frame_number.shape[0]

        pose_seq = []
        down_sample = fps // ex_fps

        betas = betas[:10]

        indices = torch.arange(0, frame_number, down_sample, device=betas.device)

        poses = poses[indices]
        root_orient = poses[:, :3]  # controls the global root orientation
        pose_body = poses[:, 3:66]
        pose_hand = poses[:, 66:]  # controls the finger articulation

        betas = repeat(betas, "c -> b c", b=poses.size(0))  # controls the body shape
        trans = trans[indices]

        if gender == "male":
            bm = self.male_bm
        else:
            bm = self.female_bm

        body = bm(
            pose_body=pose_body,
            pose_hand=pose_hand,
            betas=betas,
            root_orient=root_orient,
        )
        pose_seq = body.Jtr + rearrange(trans, "t c -> t 1 c")

        trans_matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            device=pose_seq.device,
        )
        pose_seq = pose_seq @ trans_matrix

        return pose_seq


def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def save_raw(source_path, new_name, start_frame, end_frame, fps, save_dir):
    data = np.load(source_path)
    if "humanact12" not in source_path:
        if "Eyes_Japan_Dataset" in source_path:
            data = data[3 * fps :]
        if "MPI_HDM05" in source_path:
            data = data[3 * fps :]
        if "TotalCapture" in source_path:
            data = data[1 * fps :]
        if "MPI_Limits" in source_path:
            data = data[1 * fps :]
        if "Transitions_mocap" in source_path:
            data = data[int(0.5 * fps) :]
        data = data[start_frame:end_frame]
        data[..., 0] *= -1

    data_m = swap_left_right(data)
    #     save_path = pjoin(save_dir, )
    os.makedirs(save_dir, exist_ok=True)
    np.save(osp.join(save_dir, new_name), data)
    np.save(osp.join(save_dir, "M" + new_name), data_m)
