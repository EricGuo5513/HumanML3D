import os
import os.path as osp
import multiprocessing as mp

from tqdm import tqdm

import numpy as np
import torch
from einops import rearrange, repeat

import pandas as pd

from humanml3d.body_models import BodyModel
from humanml3d.utils import extract_smpl, extract_amass, extract_humanact12


def amass_to_pose(src_path, save_path, male_bm, female_bm, ex_fps=20, device="cpu"):
    bdata = np.load(src_path, allow_pickle=True)

    fps = int(bdata.get("mocap_framerate", 0))
    frame_number = bdata.get("trans", None)
    if fps == 0 or frame_number is None:
        return 0
    else:
        frame_number = frame_number.shape[0]

    if bdata["gender"] == "male":
        bm = male_bm
    else:
        bm = female_bm

    pose_seq = []
    down_sample = fps // ex_fps

    def as_tensor(key):
        return torch.tensor(bdata[key], dtype=torch.float32, device=device)

    keys = ["poses", "betas", "trans"]
    bdata = {key: as_tensor(key) for key in keys}
    bdata["betas"] = bdata["betas"][:10]

    indices = torch.arange(0, frame_number, down_sample, device=device)

    with torch.no_grad():
        poses = bdata["poses"][indices]
        root_orient = poses[:, :3]  # controls the global root orientation
        pose_body = poses[:, 3:66]
        pose_hand = poses[:, 66:]  # controls the finger articulation

        betas = repeat(
            bdata["betas"], "c -> b c", b=poses.size(0)
        )  # controls the body shape
        trans = bdata["trans"][indices]
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
        pose_seq_np = (pose_seq @ trans_matrix).cpu().numpy()

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    np.save(save_path, pose_seq_np)
    return fps


def extract_files(data_dir, amass_root, smpl_root, pose_dir, workers=None):
    for root in [amass_root, smpl_root]:
        os.makedirs(root, exist_ok=True)

    extract_humanact12(data_dir, pose_dir)
    extract_smpl(data_dir, smpl_root, workers)
    extract_amass(data_dir, amass_root, workers)


def process_raw(amass_root, smpl_root, save_root, ex_fps, device):
    male_bm_path = osp.join(smpl_root, "smplh/male/model.npz")
    male_dmpl_path = osp.join(smpl_root, "dmpls/male/model.npz")

    female_bm_path = osp.join(smpl_root, "smplh/female/model.npz")
    female_dmpl_path = osp.join(smpl_root, "dmpls/female/model.npz")

    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    male_bm = BodyModel(
        bm_fname=male_bm_path,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=male_dmpl_path,
    ).to(device)

    female_bm = BodyModel(
        bm_fname=female_bm_path,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=female_dmpl_path,
    ).to(device)

    paths = []
    dataset_names = []
    for root, dirs, files in os.walk(amass_root):
        for name in files:
            if name.split(".")[-1] != "npz":
                continue

            dataset_name = osp.basename(root)
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            paths.append(os.path.join(root, name))

    for path in tqdm(paths):
        save_path = path.replace(amass_root, save_root)
        save_path = save_path[:-3] + "npy"
        amass_to_pose(path, save_path, male_bm, female_bm, ex_fps, device)


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


def segment_mirror_and_relocate(save_root, index_path="index.csv", save_dir="joints"):
    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    fps = 20

    args = []
    for i in range(total_amount):
        source_path = index_file.loc[i]["source_path"]
        source_path = osp.normpath(source_path)
        source_path = source_path.replace("pose_data", save_root)

        new_name = index_file.loc[i]["new_name"]
        start_frame = index_file.loc[i]["start_frame"]
        end_frame = index_file.loc[i]["end_frame"]

        args.append((source_path, new_name, start_frame, end_frame, fps, save_dir))

    with mp.Pool() as p:
        p.starmap(save_raw, args)
