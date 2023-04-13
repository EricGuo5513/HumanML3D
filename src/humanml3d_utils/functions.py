from typing import Tuple, Iterable
from pathlib import Path
import multiprocessing as mp
import itertools
import functools

import pandas as pd
import numpy as np
import torch

from .core import io
from .core.raw_pose_processing import swap_left_right
from .core.motion_representation import process_file
from .core.amass_body_model import AMASSBodyModel
from .core.skeletons import Skeleton, skeleton_factory


def extract_amass_files(smpl_dir: Path, dst: Path):
    with mp.Pool() as p:
        for paths in p.imap_unordered(
            functools.partial(io.extract_tar, dst=dst),
            [smpl_dir / name for name in io.amass_files],
        ):
            for path in paths:
                if path.suffix == ".npz":
                    yield path


def extract_smpl_files(smpl_dir: Path, dst: Path):
    with mp.Pool() as p:
        smpl_paths, dmpl_paths = list(
            p.starmap(
                io.extract_tar,
                (
                    (smpl_dir / f"{name}.tar.xz", dst / name)
                    for name in ("smplh", "dmpls")
                ),
            )
        )

    for path in itertools.chain(smpl_paths, dmpl_paths):
        if "smplh/male" in str(path):
            male_bm_path = path
        if "smplh/female" in str(path):
            female_bm_path = path
        if "dmpls/male" in str(path):
            male_dmpl_path = path
        if "dmpls/female" in str(path):
            female_dmpl_path = path

    return male_bm_path, female_bm_path, male_dmpl_path, female_dmpl_path


def extract_humanact12(path: Path, dst: Path):
    for path in io.extract_zip(path, dst):
        if path.suffix == ".npy":
            yield np.load(path), path


def positions(paths: Iterable[Path], body_model: AMASSBodyModel, device):
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

        yield pose_seq, path.with_suffix(".npy")


def format(
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


def load_splits(splits_path: Path, joint_dir: Path, text_dir: Path):
    with open(splits_path) as f:
        splits = f.read().splitlines()
    for name in splits:
        joint_path = joint_dir / f"{name}.npy"
        text_path = text_dir / f"{name}.txt"
        yield joint_path, text_path
