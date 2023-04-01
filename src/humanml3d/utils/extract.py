import os
import os.path as osp

import multiprocessing as mp

import zipfile
import tarfile

from .path import add_root

tar_files = [
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

zip_files = ["humanact12.zip"]


def find_missing(paths_to_search):
    paths_to_search = set(paths_to_search)
    missing_paths = set(paths_to_search)

    for path in paths_to_search:
        if osp.exists(path):
            missing_paths.remove(path)

    return missing_paths


def raise_missing_error(missing_paths):
    if len(missing_paths) > 0:
        msg = ", ".join(list(missing_paths))
        raise FileNotFoundError(f"missing {msg}")


def get_manifest_path(path, dst):
    file_name = f"{osp.basename(path).split('.')[0]}.txt"
    manifest_path = osp.join(dst, file_name)
    return manifest_path


def read_manifest(path, dst):
    manifest_path = get_manifest_path(path, dst)

    if not osp.exists(manifest_path):
        return None

    with open(manifest_path, "r") as f:
        file_paths = f.read().splitlines()
    return file_paths


def save_manifest(path, dst):
    manifest_path = get_manifest_path(path, dst)

    paths = []
    for root, dirs, files in os.walk(dst):
        root = root.replace(dst, ".")
        for name in files:
            paths.append(osp.join(root, name))

    manifest_path = get_manifest_path(path, dst)
    os.makedirs(osp.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        f.write("\n".join(paths) + "\n")


def normalize(paths):
    return [osp.normpath(path) for path in paths]


def extracted(path, dst):
    file_paths = read_manifest(path, dst)
    if file_paths is None:
        return False

    file_paths = normalize(add_root(dst, file_paths))

    missing_paths = find_missing(file_paths)

    if len(missing_paths) > 0:
        return False
    else:
        return True


def extract_tar(path, dst, ext):
    if not extracted(path, dst):
        with tarfile.open(path, f"r:{ext}") as f:
            f.extractall(dst)

        save_manifest(path, dst)


def extract_humanact12(src, dst):
    zip_paths = add_root(src, zip_files)
    missing_paths = find_missing(zip_paths)
    raise_missing_error(missing_paths)

    for path in zip_paths:
        if not extracted(path, dst):
            with zipfile.ZipFile(path) as f:
                f.extractall(dst)

        save_manifest(path, dst)


def extract_amass(src, dst, workers=None):
    tar_paths = add_root(src, tar_files)
    missing_paths = find_missing(tar_paths)
    raise_missing_error(missing_paths)

    args = []
    for file_name in tar_files:
        path = osp.join(src, file_name)
        arg = (path, dst, "bz2")
        args.append(arg)

    if workers == 0:
        for arg in args:
            extract_tar(*arg)
    else:
        with mp.Pool(workers) as p:
            p.starmap(extract_tar, args)


smpl_files = ["smplh.tar.xz", "dmpls.tar.xz"]


def extract_smpl(src, dst, workers=None):
    smpl_paths = [osp.join(src, file) for file in smpl_files]
    missing_paths = find_missing(smpl_paths)
    raise_missing_error(missing_paths)

    args = []
    for file_name in smpl_files:
        path = osp.join(src, file_name)
        arg = (path, osp.join(dst, file_name.split(".")[0]), "xz")
        args.append(arg)

    if workers == 0:
        for arg in args:
            extract_tar(*arg)
    else:
        with mp.Pool(workers) as p:
            p.starmap(extract_tar, args)
