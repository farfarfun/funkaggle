"""覆盖 feature.py / download.py 的可执行公开 API。"""

from pathlib import Path

import pytest
import cv2
import numpy as np

from funkaggle.deepfake.download import get_url
from funkaggle.deepfake import feature as feature_mod
from funkaggle.deepfake.feature import video2img_file


def test_get_url_uses_kaggle_without_signed_query() -> None:
    url = get_url(3)
    assert "kaggle.com/api/v1/competitions/data/download" in url
    assert "GoogleAccessId" not in url
    assert url.endswith("dfdc_train_part_03.zip")


def test_get_url_rejects_invalid_index() -> None:
    with pytest.raises(IndexError):
        get_url(50)

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


def test_video2img_file_writes_detected_face(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """检测到人脸时，应裁剪并写出抽取帧。"""
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    class FakeCapture:
        def get(self, _property: int) -> int:
            return 1

        def read(self) -> tuple[bool, np.ndarray]:
            return True, frame

        def release(self) -> None:
            pass

    writes: list[tuple[str, tuple[int, ...]]] = []
    monkeypatch.setattr(feature_mod.cv2, "VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr(feature_mod, "face_locations", lambda _frame: [(2, 12, 10, 4)])
    monkeypatch.setattr(
        feature_mod.cv2,
        "imwrite",
        lambda path, image: writes.append((path, image.shape)) or True,
    )

    out_dir = tmp_path / "out"
    video2img_file("sample.mp4", str(out_dir), n_frames=1, index=7)

    assert writes == [(str(out_dir / "7-1000-sample.jpg"), (8, 8, 3))]
