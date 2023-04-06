import os
import os.path as osp

import click
from tqdm import tqdm

import numpy as np
import torch

import humanml3d
from humanml3d.skeletons import Skeleton, skeleton_factory
from humanml3d.utils import add_root
import humanml3d.motion_representation as mr_utils


@click.group()
@click.option("--src", default=".")
@click.option("--dst", default=".")
@click.pass_context
def cli(ctx, src, dst):
    ctx.ensure_object(dict)

    ctx.obj["SRC"] = src
    ctx.obj["DST"] = dst


@cli.command()
@click.option("--workers", default=0)
@click.pass_context
def extract(ctx, workers):
    src = ctx.obj["SRC"]
    dst = ctx.obj["DST"]

    amass_dir, smpl_dir, pose_dir = add_root(
        dst, ["amass_root", "body_models", "pose_data"]
    )
    humanml3d.extract_files(src, amass_dir, smpl_dir, pose_dir, workers)


@cli.command()
@click.option("--fps", default=20)
@click.option("--target_id", default="default")
@click.option("--dataset", default="humanml3d")
@click.option("--enable_cuda", is_flag=True)
@click.pass_context
def preprocess(ctx, fps, target_id, dataset, enable_cuda):
    src = ctx.obj["SRC"]
    dst = ctx.obj["DST"]

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    amass_dir, smpl_dir, save_root = add_root(
        dst, ["amass_root", "body_models", "pose_data"]
    )
    humanml3d.process_raw(amass_dir, smpl_dir, save_root, fps, device)

    index_path = osp.join(src, "index.csv")

    pose_dir, joints_dir = add_root(dst, ["pose_data", "joints"])

    humanml3d.segment_mirror_and_relocate(pose_dir, index_path, joints_dir)

    if target_id == "default":
        if dataset.lower() == "kit":
            target_id = "03950"
        elif dataset.lower() == "humanml3d":
            target_id = "000021"
        else:
            raise ValueError

    # ds_num = 8
    data_dir, joint_dir, joint_vec_dir = add_root(
        dst, ["joints", "new_joints", "new_joint_vecs"]
    )

    os.makedirs(joint_dir, exist_ok=True)
    os.makedirs(joint_vec_dir, exist_ok=True)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() and enable_cuda else "cpu"
    )
    raw_offsets, kinematic_chain = skeleton_factory(dataset, device)
    skeleton = Skeleton(raw_offsets, kinematic_chain, "cpu")

    target_skeleton = np.load(osp.join(data_dir, target_id + ".npy"))
    target_skeleton = target_skeleton.reshape(len(target_skeleton), -1, 3)
    target_skeleton = torch.from_numpy(target_skeleton).to(device)

    target_offsets = skeleton.get_offsets_joints(target_skeleton[0])

    source_list = os.listdir(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(osp.join(data_dir, source_file))[:, : skeleton.njoints()]

        if source_data.shape[0] == 1:
            print(source_file)
            continue

        data, ground_positions, positions, l_velocity = mr_utils.process_file(
            raw_offsets, kinematic_chain, source_data, 0.002, target_offsets
        )
        rec_ric_data = mr_utils.recover_from_ric(
            torch.from_numpy(data).unsqueeze(0).float().to(device),
            skeleton.njoints(),
        )
        np.save(osp.join(joint_dir, source_file), rec_ric_data.squeeze().cpu().numpy())
        np.save(osp.join(joint_vec_dir, source_file), data)
        frame_num += data.shape[0]

    src = ctx.obj["SRC"]
    dst = ctx.obj["DST"]

    joint_path, joint_vec_path = add_root(dst, ["new_joints", "new_joint_vecs"])

    # The given data is used to double check if you are on the right track.
    ref_joint = np.load(osp.join(src, "new_joints/012314.npy"))
    ref_joint_vec = np.load(osp.join(src, "new_joint_vecs/012314.npy"))

    joint = np.load(osp.join(joint_path, "012314.npy"))
    joint_vec = np.load(osp.join(joint_vec_path, "012314.npy"))

    click.echo(abs(ref_joint - joint).sum())
    click.echo(abs(ref_joint_vec - joint_vec).sum())

    if dataset.lower() == "humanml3d":
        num_joints = 22
    else:
        raise NotImplementedError

    mean, std = humanml3d.mean_variance(joint_vec_path, num_joints)
    mean_path, std_path = add_root(dst, ["Mean.npy", "Std.npy"])
    np.save(mean_path, mean)
    np.save(std_path, std)

    ref_mean_path, ref_std_path = add_root(src, ["Mean.npy", "Std.npy"])
    ref_mean = np.load(ref_mean_path)
    ref_std = np.load(ref_std_path)

    click.echo(np.abs(mean - ref_mean).sum())
    click.echo(np.abs(std - ref_std).sum())


if __name__ == "__main__":
    cli()
