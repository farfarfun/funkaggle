"""覆盖配置解析的优先级：命令行参数 > 环境变量 > 默认值。"""

from pathlib import Path

import pytest

from funkaggle.deepfake.config import (
    DEFAULT_DATA_ROOT,
    ENV_DATA_ROOT,
    build_config,
    resolve_data_root,
)


def test_resolve_data_root_uses_default_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_DATA_ROOT, raising=False)
    assert resolve_data_root(None) == DEFAULT_DATA_ROOT


def test_resolve_data_root_prefers_env_over_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_DATA_ROOT, str(tmp_path))
    assert resolve_data_root(None) == tmp_path.resolve()


def test_resolve_data_root_prefers_cli_over_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_DATA_ROOT, "/should/not/be/used")
    cli_dir = tmp_path / "cli"
    assert resolve_data_root(str(cli_dir)) == cli_dir.resolve()


def test_build_config_derives_paths(tmp_path: Path) -> None:
    config = build_config(str(tmp_path))
    assert config.data_root == tmp_path.resolve()
    assert config.train_data == config.data_root / "train_data"
    assert config.test_data == config.data_root / "test_data"
    assert config.predict_data == config.data_root / "predict_data"
    assert config.submission_path == config.data_root / "result" / "submission.csv"
