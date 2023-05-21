from typing import Tuple, Iterable, BinaryIO

from pathlib import Path
import multiprocessing as mp
import functools

import tarfile

import pandas as pd
import numpy as np
import torch

from .core import extract
from .core.raw_pose_processing import swap_left_right
from .core.motion_representation import process_file, recover_from_ric
from .core.amass_body_model import AMASSBodyModel
from .core.skeletons import Skeleton, skeleton_factory

import os

torch.set_num_threads(os.cpu_count())

amass_files = [
    "ACCAD.tar.bz2",
    "BMLhandball.tar.bz2",
    "BMLmovi.tar.bz2",
    "BMLrub.tar.bz2",
    "CMU.tar.bz2",
    "DFaust.tar.bz2",
    "EKUT.tar.bz2",
    "EyesJapanDataset.tar.bz2",
    "HDM05.tar.bz2",
    "HumanEva.tar.bz2",
    "KIT.tar.bz2",
    "MoSh.tar.bz2",
    "PosePrior.tar.bz2",
    "SFU.tar.bz2",
    "SSM.tar.bz2",
    "TCDHands.tar.bz2",
    "TotalCapture.tar.bz2",
    "Transitions.tar.bz2",
]

humanact_file = "humanact12.zip"

smpl_files = {"smpl": "smplh.tar.xz", "dmpl": "dmpls.tar.xz"}


def extract_tar_files(tar_path: Path, dst: Path, suffix: str):
    paths = []
    for path in extract(tar_path, dst):
        if path.suffix == suffix:
            paths.append(path)
    return paths


def extract_smpl_files(smpl_dir: Path, dst: Path):
    for file in ["smplh.tar.xz", "dmpls.tar.xz"]:
        with tarfile.open(smpl_dir / file, "r:xz") as tar:
            path = dst / file.split(".")[0]
            tar.extractall(path)
        yield path


def extract_amass_files(amass_dir: Path, dst: Path):
    amass_paths = [amass_dir / file for file in amass_files]

    all_paths = []
    with mp.Pool() as p:
        for paths in p.imap_unordered(
            functools.partial(extract_tar_files, dst=dst, suffix=".npz"), amass_paths
        ):
            for path in paths:
                all_paths.append(path)

    return all_paths


def load_humanact12(path: Path, dst: Path):
    array_path_pairs = []
    for path in extract(path, dst):
        if path.suffix == ".npy":
            array_path_pairs.append((np.load(path), path))
    return array_path_pairs


def to_positions(
    paths: Iterable[Tuple[BinaryIO, Path]], body_model: AMASSBodyModel, device
):
    array_path_pairs = []
    for path in paths:
        bdata = np.load(path, allow_pickle=True)

        fps = int(bdata.get("mocap_framerate", 0))
        frame_number = bdata.get("trans", None)
        if fps == 0 or frame_number is None:
            continue
        else:
            frame_number = frame_number.shape[0]

        gender = bdata["gender"]

        def as_tensor(key):
            return torch.tensor(bdata[key], dtype=torch.float32, device=device)

        keys = ["poses", "betas", "trans"]
        bdata = {key: as_tensor(key) for key in keys}

        poses = bdata["poses"]
        betas = bdata["betas"]
        trans = bdata["trans"]

        with torch.inference_mode():
            pose_seq = body_model(trans, gender, fps, poses, betas).cpu().numpy()

        array_path_pairs.append((pose_seq, path.with_suffix(".npy")))

    return array_path_pairs


def format_poses(
    array_path_pairs: Iterable[Tuple[np.ndarray, Path]],
    root: Path,
    index_path: Path,
    fps: int = 20,
):
    df = pd.read_csv(index_path)

    for array, path in array_path_pairs:
        query = "./" + str("pose_data" / path.relative_to(root))

        if (df["source_path"] == query).any():
            index = df.index[df["source_path"] == query]

            for i in index:
                start_frame, end_frame, new_name = (
                    df.iloc[i]["start_frame"],
                    df.iloc[i]["end_frame"],
                    df.iloc[i]["new_name"],
                )

                new_array = array.copy()

                if "humanact12" not in str(path):
                    if "Eyes_Japan_Dataset" in str(path):
                        new_array = new_array[3 * fps :]
                    if "MPI_HDM05" in str(path):
                        new_array = new_array[3 * fps :]
                    if "TotalCapture" in str(path):
                        new_array = new_array[1 * fps :]
                    if "MPI_Limits" in str(path):
                        new_array = new_array[1 * fps :]
                    if "Transitions_mocap" in str(path):
                        new_array = new_array[int(0.5 * fps) :]

                    new_array = new_array[start_frame:end_frame]
                    new_array[..., 0] *= -1

                path = path.with_name(new_name).with_suffix(".npy")
                yield new_array, path


def flip_left_right(array_path_pairs: Iterable[Tuple[np.ndarray, Path]]):
    for array, path in array_path_pairs:
        yield array, path

        flipped_array: np.ndarray = swap_left_right(array)
        flipped_path = path.with_name("M" + path.name)

        yield flipped_array, flipped_path


def motion_representation(
    array_path_pairs: Iterable[Tuple[np.ndarray, Path]], device="cpu"
):
    raw_offsets, kinematic_chain = skeleton_factory("humanml3d", device)
    skeleton = Skeleton(raw_offsets, kinematic_chain, "cpu")

    array_path_pairs = list(array_path_pairs)

    target_skeleton = None
    for array, path in array_path_pairs:
        if "000021.npy" in str(path):
            target_skeleton = array
            break

    assert target_skeleton is not None

    target_skeleton = target_skeleton.reshape(len(target_skeleton), -1, 3)
    target_skeleton = torch.from_numpy(target_skeleton).cpu()

    target_offsets = skeleton.get_offsets_joints(target_skeleton[0])

    for array, path in array_path_pairs:
        if array.shape[0] == 1:
            print(f"skipping {path}")
            continue

        array = array[:, : skeleton.njoints()]

        data, ground_positions, positions, l_velocity = process_file(
            raw_offsets, kinematic_chain, array, 0.002, target_offsets
        )

        yield data, path


def recover_from_representation(array_path_pairs: Iterable[Tuple[np.ndarray, Path]]):
    for array, path in array_path_pairs:
        array = recover_from_ric(
            torch.from_numpy(array).unsqueeze(0).float(), joints_num=22
        )

        yield array, path


def load_splits(splits_path: Path, joint_dir: Path, text_dir: Path):
    with open(splits_path) as f:
        splits = f.read().splitlines()
    for name in splits:
        joint_path = joint_dir / f"{name}.npy"
        text_path = text_dir / f"{name}.txt"
        yield joint_path, text_path


def compute_stats(motions, num_joints):
    count = 0
    mean = 0
    for array in motions:
        count += array.shape[0]
        mean += array.sum(0)
    mean /= count

    variance = 0
    for array in motions:
        variance += ((array - mean) ** 2).sum(0)
    std = np.sqrt(variance / count)

    std[0:1] = std[0:1].mean() / 1.0
    std[1:3] = std[1:3].mean() / 1.0
    std[3:4] = std[3:4].mean() / 1.0
    std[4 : 4 + (num_joints - 1) * 3] = std[4 : 4 + (num_joints - 1) * 3].mean() / 1.0
    std[4 + (num_joints - 1) * 3 : 4 + (num_joints - 1) * 9] = (
        std[4 + (num_joints - 1) * 3 : 4 + (num_joints - 1) * 9].mean() / 1.0
    )
    std[4 + (num_joints - 1) * 9 : 4 + (num_joints - 1) * 9 + num_joints * 3] = (
        std[4 + (num_joints - 1) * 9 : 4 + (num_joints - 1) * 9 + num_joints * 3].mean()
        / 1.0
    )
    std[4 + (num_joints - 1) * 9 + num_joints * 3 :] = (
        std[4 + (num_joints - 1) * 9 + num_joints * 3 :].mean() / 1.0
    )

    assert 8 + (num_joints - 1) * 9 + num_joints * 3 == std.shape[-1]

    return mean, std
