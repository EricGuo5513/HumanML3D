from pathlib import Path

from io import BytesIO

import zipfile
import tarfile


def extract_zip(path: Path, dst: Path):
    with zipfile.ZipFile(path) as zip_file:
        zip_file.extractall(dst)
        infos = zip_file.infolist()

    for info in infos:
        if not info.is_dir():
            yield dst / info.filename


def extract(path: Path, dst: Path):
    ext = path.suffix.split(".")[-1]

    if ext == "zip":
        yield from extract_zip(path, dst)
    else:
        with tarfile.open(path, f"r:{ext}") as tar:
            tar.extractall(dst)

            members = tar.getmembers()

        for member in members:
            if member.isfile():
                yield dst / member.name
