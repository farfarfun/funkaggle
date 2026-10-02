"""通过 Kaggle 官方接口下载并解压 DFDC 训练数据分卷。"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from funfile.compress.zipfile import ZipFile
from funget import download


def get_url(file_index: int = 0) -> str:
    """返回指定分卷的 Kaggle 官方下载地址。

    认证由 Kaggle 客户端环境（`~/.kaggle/kaggle.json` 或环境变量）提供，
    地址本身不包含任何凭据或签名参数。

    Args:
        file_index: 分卷编号，取值范围 0-49。

    Returns:
        形如 `.../download/deepfake-detection-challenge/dfdc_train_part_XX.zip`
        的下载地址。

    Raises:
        IndexError: `file_index` 不在 0-49 范围内。
    """
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
    """下载单个分卷，可选自动解压。

    Args:
        save_dir: 保存目录，不存在则自动创建；为空时使用安全默认目录
            `~/.cache/funkaggle/deepfake`。
        file_index: 分卷编号，取值范围 0-49。
        unzip: 下载完成后是否自动解压。

    Raises:
        IndexError: `file_index` 不在 0-49 范围内（见 `get_url`）。
        RuntimeError: `funget.download` 返回下载失败。
    """
    target_dir = Path(save_dir or Path.home() / ".cache/funkaggle/deepfake")
    target_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"dfdc_train_part_{file_index:02}.zip"
    save_path = target_dir / file_name
    url = get_url(file_index)
    ok = download(url, os.fspath(save_path))
    if not ok:
        raise RuntimeError(
            f"下载失败: url={url}, save_path={save_path}, file_index={file_index}"
        )
    if unzip:
        unzip_file(os.fspath(save_path), os.fspath(target_dir))


def unzip_file(save_path: str, save_dir: str) -> None:
    """安全解压单个分卷 zip 文件到 save_dir。

    解压前校验压缩包存在且合法；解压时逐个检查成员的解析后目标路径必须
    落在 `save_dir` 内，拒绝绝对路径、`..` 等试图路径穿越的成员，避免恶意
    或被替换的压缩包写出目标目录。

    Args:
        save_path: zip 文件路径。
        save_dir: 解压目标目录，解压后所有文件必须位于该目录内。

    Raises:
        FileNotFoundError: `save_path` 不存在。
        RuntimeError: `save_path` 不是合法 zip 文件，或其中存在路径穿越成员。
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"待解压文件不存在: {save_path}")

    root = os.path.realpath(save_dir)
    try:
        archive = ZipFile(save_path)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"不是合法的 zip 文件: {save_path}") from exc

    with archive:
        for member in archive.namelist():
            target = os.path.realpath(os.path.join(root, member))
            if os.path.commonpath([root, target]) != root:
                raise RuntimeError(
                    f"zip 成员路径越出解压目录，拒绝解压: {save_path}!{member}"
                )
        archive.extractall(root)


def download_files(
    save_dir: str, file_index_list: list[int], unzip: bool = True
) -> None:
    """批量下载多个分卷。

    Args:
        save_dir: 保存目录，透传给 `download_file`。
        file_index_list: 待下载的分卷编号列表。
        unzip: 下载完成后是否自动解压。

    Raises:
        IndexError: 列表中某个编号不在 0-49 范围内。
        RuntimeError: 任一分卷下载失败，中止后续下载。
    """
    for index in file_index_list:
        download_file(save_dir, index, unzip=unzip)
