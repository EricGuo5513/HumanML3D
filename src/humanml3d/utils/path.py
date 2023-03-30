import os.path as osp


def add_root(root, paths):
    return [osp.join(root, path) for path in paths]
