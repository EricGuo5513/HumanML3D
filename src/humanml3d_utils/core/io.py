from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import zipfile
import tarfile

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

smpl_files = ["smplh.tar.xz", "dmpls.tar.xz"]


def load_npy(directory: Path):
    for path in directory.glob(".npy"):
        yield np.load(path), path


def load_npz(directory: Path):
    for path in directory.glob(".npz"):
        yield np.load(path), path


def save_npy(array_path_pair):
    for array, path in array_path_pair:
        yield np.save(path, array)


def extract_zip(path: Path, dst: Path):
    paths = []

    with zipfile.ZipFile(path) as f:
        f.extractall(dst)

        for info in f.infolist():
            if not info.is_dir():
                paths.append(dst / info.filename)

    return paths


def extract_tar(path: Path, dst: Path):
    paths = []

    ext = path.suffix.split(".")[-1]
    with tarfile.open(path, f"r:{ext}") as tar:
        tar.extractall(dst)

        for member in tar.getmembers():
            if member.isfile():
                paths.append(dst / member.name)

    return paths


class IndexFile:
    metadata: Dict[Path, Tuple[int, int, Path]]

    def __init__(self, path: Path) -> None:
        df = pd.read_csv(path)
        breakpoint()
        df = df.to_dict("list")

        metadata = {}
        for path, start, end, name in zip(
            df["source_path"], df["start_frame"], df["end_frame"], df["new_name"]
        ):
            path = Path(path).relative_to("pose_data")
            if path in metadata.keys():
                print(path)
            metadata[path] = (start, end, name)

        self.metadata = metadata

    def exists(self, path: Path):
        return path in self.metadata.keys()

    def index(self, path: Path):
        start, end, name = self.metadata[path]
        return start, end, name
