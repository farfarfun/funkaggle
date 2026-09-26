"""覆盖 feature.py / download.py 的可执行公开 API。

这两个模块依赖 tensorflow / face_recognition（含 dlib 编译）等重依赖，
审计所用沙箱磁盘紧张未安装，故用 importorskip 优雅跳过；在装有完整
依赖（见 pyproject.toml）的环境或 CI 中会实际执行。
"""

from pathlib import Path

import pytest

from funkaggle.deepfake.download import get_url


def test_get_url_uses_kaggle_without_signed_query() -> None:
    url = get_url(3)
    assert "kaggle.com/api/v1/competitions/data/download" in url
    assert "GoogleAccessId" not in url
    assert url.endswith("dfdc_train_part_03.zip")


def test_get_url_rejects_invalid_index() -> None:
    with pytest.raises(IndexError):
        get_url(50)

cv2 = pytest.importorskip("cv2")
pytest.importorskip("face_recognition")
np = pytest.importorskip("numpy")

from funkaggle.deepfake.feature import video2img_file


def test_video2img_file_skips_video_without_faces(tmp_path: Path) -> None:
    """纯色帧里检测不到人脸时，不应生成任何图片（边界路径）。"""
    video_path = tmp_path / "blank.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5, (64, 64))
    for _ in range(5):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    out_dir = tmp_path / "out"
    video2img_file(str(video_path), str(out_dir), n_frames=2, index=1)

    assert not out_dir.exists() or not any(out_dir.iterdir())
