import os
import os.path as osp

from tqdm import tqdm

import numpy as np

import torch

from einops import rearrange, repeat

from humanml3d.body_models import BodyModel
from humanml3d.utils.extract import extract_smpl_files, extract_zip_files

comp_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


def amass_to_pose(src_path, save_path, male_bm, female_bm, ex_fps=20):
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
        return torch.tensor(bdata[key], dtype=torch.float32, device=comp_device)

    keys = ["poses", "betas", "trans"]
    bdata = {key: as_tensor(key) for key in keys}
    bdata["betas"] = bdata["betas"][:10]

    indices = torch.arange(0, frame_number, down_sample, device=comp_device)

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
            device=comp_device,
        )
        pose_seq_np = (pose_seq @ trans_matrix).cpu().numpy()

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    np.save(save_path, pose_seq_np)
    return fps


def extract_files(data_dir, amass_root, smpl_root, workers=None):
    for root in [amass_root, smpl_root, save_root]:
        os.makedirs(root, exist_ok=True)

    extract_smpl_files(data_dir, smpl_root, workers)
    extract_zip_files(data_dir, amass_root, workers)


def process_raw(amass_root, smpl_root, save_root):
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
    ).to(comp_device)

    female_bm = BodyModel(
        bm_fname=female_bm_path,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=female_dmpl_path,
    ).to(comp_device)

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
        amass_to_pose(path, save_path, male_bm, female_bm)


if __name__ == "__main__":
    data_dir = "amass"
    amass_root = "/data/humanml3d/amass_root"
    smpl_root = "/data/humanml3d/body_models"
    save_root = "/data/humanml3d/pose_data"

    extract_files(data_dir, amass_root, smpl_root, workers=None)

    process_raw(amass_root, smpl_root, save_root)
