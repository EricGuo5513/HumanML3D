import os

import zipfile

from jsonargparse import CLI

import numpy as np
import torch

from pathlib import Path
import humanml3d_utils
from humanml3d_utils.core import AMASSBodyModel


def process(data_dir: Path, motion_dir: Path, text_dir: Path):
    extract_dir = data_dir / "raw"

    smpl_path, dmpl_path = humanml3d_utils.extract_smpl_files(data_dir, extract_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    body_model = AMASSBodyModel(smpl_path, dmpl_path).to(device)

    amass_paths = humanml3d_utils.extract_amass_files(data_dir, extract_dir)
    humanact_paths = humanml3d_utils.load_humanact12(
        data_dir / "humanact12.zip", extract_dir
    )

    positions = humanml3d_utils.to_positions(
        amass_paths,
        body_model,
        device,
    )
    positions.extend(humanact_paths)

    pose_representation = humanml3d_utils.motion_representation(
        humanml3d_utils.flip_left_right(
            humanml3d_utils.format_poses(
                positions, root=extract_dir, index_path=data_dir / "index.csv", fps=20
            )
        )
    )

    for array, path in pose_representation:
        path = motion_dir / path.name
        os.makedirs(path.parent, exist_ok=True)
        np.save(path, array)

    with zipfile.ZipFile(data_dir / "texts.zip") as f:
        f.extractall(text_dir)


def validate(motion_dir: Path, text_dir: Path, split_path: Path):
    path = motion_dir / "012314.npy"
    representation = np.load(path)
    ((joints, _),) = humanml3d_utils.recover_from_representation(
        [
            (representation, path),
        ]
    )

    sample_representation = np.load("samples/new_joint_vecs/012314.npy")
    sample_joints = np.load("samples/new_joints/012314.npy")

    diff = abs(representation - sample_representation).sum()
    assert diff < 0.02, diff
    diff = abs(joints - sample_joints).sum()
    assert diff < 0.02, diff

    paths = humanml3d_utils.load_splits(split_path, motion_dir, text_dir)

    motions = []
    for motion_path, text_path in paths:
        array = np.load(motion_path)
        motions.append(array)

    mean, std = humanml3d_utils.compute_stats(motions, num_joints=22)

    sample_mean = np.load("samples/Mean.npy")
    diff = abs(mean - sample_mean).sum()
    assert diff < 1, diff

    sample_std = np.load("samples/Std.npy")
    diff = abs(std - sample_std).sum()
    assert diff < 1, diff


if __name__ == "__main__":
    CLI()
