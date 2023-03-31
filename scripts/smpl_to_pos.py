import os
import os.path as osp

import click
from tqdm import tqdm

import numpy as np
import torch

import humanml3d
from humanml3d.skeletons import skeleton_factory
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

    amass_dir, smpl_dir = add_root(dst, ["amass_root", "body_models"])
    humanml3d.extract_files(src, amass_dir, smpl_dir, workers)


@cli.command()
@click.option("--fps", default=20)
@click.pass_context
def process(ctx, fps):
    dst = ctx.obj["DST"]

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    amass_dir, smpl_dir, save_root = add_root(
        dst, ["amass_root", "body_models", "pose_data"]
    )
    humanml3d.process_raw(amass_dir, smpl_dir, save_root, fps, device)


@cli.command()
@click.pass_context
def augment(ctx):
    src = ctx.obj["SRC"]
    dst = ctx.obj["DST"]

    index_path = osp.join(src, "index.csv")

    pose_dir, joints_dir = add_root(dst, ["pose_data", "joints"])

    humanml3d.segment_mirror_and_relocate(pose_dir, index_path, joints_dir)


@cli.command()
@click.option("--target_id", default="default")
@click.option("--dataset", default="humanml3d")
@click.option("--enable_cuda", is_flag=True)
@click.pass_context
def pos_to_mmm(ctx, target_id, dataset, enable_cuda):
    dst = ctx.obj["DST"]

    if target_id == "default":
        if dataset.lower() == "kit":
            target_id = "03950"
        elif dataset.lower() == "humanml3d":
            target_id = "0000021"
        else:
            raise ValueError

    example_id = "000021"
    joints_num = 22
    # ds_num = 8
    data_dir, save_dir1, save_dir2 = add_root(
        dst, ["joints", "new_joints", "new_joint_vecs"]
    )

    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() and enable_cuda else "cpu"
    )
    skeleton = skeleton_factory(dataset, device)

    target_skeleton = np.load(osp.join(data_dir, example_id + ".npy"))
    target_skeleton = target_skeleton.reshape(len(target_skeleton), -1, 3)
    target_skeleton = torch.from_numpy(target_skeleton).to(device)

    target_offsets = skeleton.get_offsets_joints(target_skeleton[0])

    source_list = os.listdir(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(osp.join(data_dir, source_file))[:, : skeleton.njoints()]
        try:
            data, ground_positions, positions, l_velocity = mr_utils.process_file(
                skeleton, source_data, 0.002, target_offsets
            )
            rec_ric_data = mr_utils.recover_from_ric(
                torch.from_numpy(data).unsqueeze(0).float(), joints_num
            )
            np.save(osp.join(save_dir1, source_file), rec_ric_data.squeeze().numpy())
            np.save(osp.join(save_dir2, source_file), data)
            frame_num += data.shape[0]
        except Exception as e:
            click.echo(source_file)
            click.echo(e)
    #         print(source_file)
    #         break

    click.echo(
        "Total clips: %d, Frames: %d, Duration: %fm"
        % (len(source_list), frame_num, frame_num / 20 / 60)
    )


@cli.command()
@click.option("--num_joints", default=22)
@click.pass_context
def verify(ctx, num_joints):
    src = ctx.obj["SRC"]
    dst = ctx.obj["DST"]

    joints_dir, save_dir1, save_dir2 = add_root(
        dst, ["joints", "new_joints", "new_joint_vecs"]
    )

    # The given data is used to double check if you are on the right track.
    reference1 = np.load(osp.join(src, "012314.npy"))
    reference2 = np.load(osp.join(src, "012314.npy"))

    reference1_1 = np.load(osp.join(save_dir1, "new_joints/012314.npy"))
    reference2_1 = np.load(osp.join(save_dir2, "new_joint_vecs/012314.npy"))

    click.echo(abs(reference1 - reference1_1).sum())
    click.echo(abs(reference2 - reference2_1).sum())

    mean, std = humanml3d.mean_variance(joints_dir, num_joints)

    reference1, reference2 = add_root(src, ["Mean.npy", "Std.npy"])

    click.echo(abs(mean - reference1).sum())
    click.echo(abs(std - reference2).sum())


if __name__ == "__main__":
    cli()
