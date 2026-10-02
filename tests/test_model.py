"""覆盖 MyModel 公开 API，训练和预测使用轻量测试替身。"""

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from funkaggle.deepfake import model as model_mod
from funkaggle.deepfake.model import MyModel


@pytest.fixture
def model(tmp_path: Path) -> MyModel:
    return MyModel(str(tmp_path), str(tmp_path / "train"), str(tmp_path / "test"))


def test_build_returns_and_caches_model(model: MyModel) -> None:
    built = model.build()

    assert model.model is built
    assert built.input_shape == (None, 150, 150, 3)
    assert built.output_shape == (None, 1)


def test_load_and_clear_checkpoint(
    monkeypatch: pytest.MonkeyPatch, model: MyModel
) -> None:
    checkpoint = Path(model.checkpoint_path)
    checkpoint.write_bytes(b"weights")
    loaded = object()
    monkeypatch.setattr(model_mod, "load_model", lambda path: loaded)

    model.load()
    assert model.model is loaded

    model.clear()
    assert not checkpoint.exists()


def test_train_compiles_and_fits_with_directory_generators(
    monkeypatch: pytest.MonkeyPatch, model: MyModel
) -> None:
    generators = [object(), object()]
    directories: list[tuple[str, int]] = []

    class FakeDataGenerator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def flow_from_directory(self, directory: str, **kwargs: object) -> object:
            directories.append((directory, int(kwargs["batch_size"])))
            return generators[len(directories) - 1]

    class FakeModel:
        def __init__(self) -> None:
            self.compile_kwargs: dict[str, object] = {}
            self.fit_kwargs: dict[str, object] = {}

        def compile(self, **kwargs: object) -> None:
            self.compile_kwargs = kwargs

        def fit(self, train_generator: object, **kwargs: object) -> None:
            assert train_generator is generators[0]
            self.fit_kwargs = kwargs

    fake_model = FakeModel()
    model.model = fake_model  # type: ignore[assignment]
    monkeypatch.setattr(model_mod, "ImageDataGenerator", FakeDataGenerator)
    monkeypatch.setattr(model_mod, "ModelCheckpoint", lambda *args, **kwargs: object())
    monkeypatch.setattr(model_mod, "TensorBoard", lambda *args, **kwargs: object())

    model.train(batch_size=4)

    assert directories == [(model.training_data_dir, 4), (model.testing_data_dir, 4)]
    assert fake_model.compile_kwargs["loss"] == "binary_crossentropy"
    assert fake_model.fit_kwargs["validation_data"] is generators[1]
    assert fake_model.fit_kwargs["epochs"] == 100


def test_predict_writes_video_level_submission(
    monkeypatch: pytest.MonkeyPatch, model: MyModel, tmp_path: Path
) -> None:
    batch = (np.zeros((2, 150, 150, 3)), np.array([1.0, 0.0]))

    class FakeGenerator:
        filenames: ClassVar[list[str]] = ["1/1-1000-alpha.jpg", "0/2-1000-alpha.jpg"]

        def __iter__(self):
            return iter([batch])

    generator = FakeGenerator()

    class FakeDataGenerator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def flow_from_directory(self, *_args: object, **_kwargs: object) -> object:
            return generator

    fake_predictor = SimpleNamespace(predict=lambda _images: np.array([[0.2], [0.6]]))
    model.model = fake_predictor  # type: ignore[assignment]
    monkeypatch.setattr(model_mod, "ImageDataGenerator", FakeDataGenerator)
    monkeypatch.setattr(model_mod, "tqdm", lambda iterable: iterable)
    submission = tmp_path / "result" / "submission.csv"

    result = model.predict(str(tmp_path / "predict"), str(submission), batch_size=2)

    expected = pd.DataFrame({"filename": ["alpha.mp4"], "label": [0.4]})
    pd.testing.assert_frame_equal(result[["filename", "label"]], expected)
    assert submission.exists()
