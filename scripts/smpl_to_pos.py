import os.path as osp

import click

import torch

import humanml3d
from humanml3d.utils import add_root


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


if __name__ == "__main__":
    cli()
