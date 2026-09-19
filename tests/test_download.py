"""覆盖 download.py 的公开 API：正常路径 + 边界（非法 index）。"""

import zipfile
from pathlib import Path

import pytest

from funkaggle.deepfake import download as download_mod


def test_get_url_returns_string_for_valid_index() -> None:
    url = download_mod.get_url(0)
    assert url.startswith("https://storage.googleapis.com/")


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
