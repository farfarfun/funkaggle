"""通过 Kaggle 官方接口下载并解压 DFDC 训练数据分卷。"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from funget import download


def get_url(file_index: int = 0) -> str:
    """返回 Kaggle 官方下载地址；认证由 Kaggle 客户端环境提供。"""
    if not 0 <= file_index < 50:
        raise IndexError(f"file_index 超出范围: {file_index}, 有效范围 0-49")
    file_name = f"dfdc_train_part_{file_index:02}.zip"
    return (
        "https://www.kaggle.com/api/v1/competitions/data/download/"
        f"deepfake-detection-challenge/{file_name}"
    )


def download_file(
    save_dir: str | None = None, file_index: int = 0, unzip: bool = True
) -> None:
    """下载单个分卷，可选自动解压。"""
    target_dir = Path(save_dir or Path.home() / ".cache/funkaggle/deepfake")
    target_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"dfdc_train_part_{file_index:02}.zip"
    save_path = target_dir / file_name
    download(get_url(file_index), os.fspath(save_path))
    if unzip:
        unzip_file(os.fspath(save_path), os.fspath(target_dir))


def unzip_file(save_path: str, save_dir: str) -> None:
    """解压单个分卷 zip 文件到 save_dir。"""
    with zipfile.ZipFile(save_path) as archive:
        archive.extractall(save_dir)


def download_files(
    save_dir: str, file_index_list: list[int], unzip: bool = True
) -> None:
    """批量下载多个分卷。"""
    for index in file_index_list:
        download_file(save_dir, index, unzip=unzip)
