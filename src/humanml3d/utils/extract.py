import os
import os.path as osp
import tarfile
import multiprocessing as mp

zipfiles = [
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


def check_for_missing_files(dir):
    missing_files = set(zipfiles)

    paths = os.listdir(dir)
    for path in paths:
        if path in zipfiles:
            missing_files.remove(path)

    return missing_files


def extract_tar(path, dst):
    with tarfile.open(path, "r:bz2") as f:
        f.extractall(dst)


def extract_zipfiles(dir, dst):
    missing_files = check_for_missing_files(dir)
    if len(missing_files) > 0:
        raise FileNotFoundError

    args = []
    for file_name in zipfiles:
        path = osp.join(dir, file_name)
        arg = (path, dst)
        args.append(arg)

    with mp.Pool() as p:
        p.starmap(extract_tar, args)
