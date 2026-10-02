"""覆盖 download.py 的公开 API：正常路径 + 边界（非法 index）。"""

import zipfile
from pathlib import Path

import pytest

from funkaggle.deepfake import download as download_mod


def test_get_url_returns_string_for_valid_index() -> None:
    url = download_mod.get_url(0)
    assert url.startswith("https://www.kaggle.com/api/v1/competitions/data/download/")


def test_get_url_rejects_out_of_range_index() -> None:
    with pytest.raises(IndexError):
        download_mod.get_url(999)
    with pytest.raises(IndexError):
        download_mod.get_url(-1)


def test_download_file_calls_funget_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """download_file 应把 (url, save_path) 转发给 funget.download，不直连网络。"""
    calls = []

    def fake_download(url: str, filepath: str, **kwargs) -> bool:
        calls.append((url, filepath))
        Path(filepath).write_bytes(b"")
        return True

    monkeypatch.setattr(download_mod, "download", fake_download)
    download_mod.download_file(save_dir=str(tmp_path), file_index=0, unzip=False)

    assert len(calls) == 1
    url, save_path = calls[0]
    assert url == download_mod.get_url(0)
    assert save_path == str(tmp_path / "dfdc_train_part_00.zip")


def test_unzip_file_extracts_into_save_dir(tmp_path: Path) -> None:
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("hello.txt", "hi")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    download_mod.unzip_file(str(zip_path), str(out_dir))

    assert (out_dir / "hello.txt").read_text() == "hi"


def test_unzip_file_rejects_path_traversal_member(tmp_path: Path) -> None:
    """zip 内含 `../` 逃逸成员时必须拒绝解压，不能写出目标目录（路径穿越防护）。"""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.txt", "pwned")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(RuntimeError, match="越出解压目录"):
        download_mod.unzip_file(str(zip_path), str(out_dir))

    assert not (tmp_path / "escape.txt").exists()


def test_unzip_file_rejects_absolute_path_member(tmp_path: Path) -> None:
    """zip 内含绝对路径成员时同样要拒绝解压。"""
    zip_path = tmp_path / "evil_abs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("/tmp/escape_abs.txt", "pwned")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(RuntimeError, match="越出解压目录"):
        download_mod.unzip_file(str(zip_path), str(out_dir))


def test_unzip_file_raises_for_missing_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        download_mod.unzip_file(str(tmp_path / "missing.zip"), str(out_dir))


def test_unzip_file_raises_for_invalid_zip(tmp_path: Path) -> None:
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"not a zip file")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(RuntimeError, match="不是合法的 zip 文件"):
        download_mod.unzip_file(str(bad_zip), str(out_dir))


def test_download_file_raises_when_download_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """funget.download 返回 False 时应中止，不能继续解压失败产物。"""
    monkeypatch.setattr(download_mod, "download", lambda url, filepath, **kwargs: False)
    with pytest.raises(RuntimeError, match="下载失败"):
        download_mod.download_file(save_dir=str(tmp_path), file_index=0)
