from pathlib import Path

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
