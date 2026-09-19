"""覆盖 run.py 的 CLI 参数解析与路径优先级派生，不触发重依赖导入。"""

from pathlib import Path

import pytest

from funkaggle.deepfake.config import ENV_DATA_ROOT
from funkaggle.deepfake.run import _build_parser, main


def test_parser_requires_action() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_rejects_unknown_action() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bogus"])


def test_parser_parses_data_root_override() -> None:
    parser = _build_parser()
    args = parser.parse_args(["download", "--data-root", "/tmp/x"])
    assert args.action == "download"
    assert args.data_root == "/tmp/x"


def test_main_dispatches_to_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(ENV_DATA_ROOT, raising=False)
    calls = []
    monkeypatch.setattr(
        "funkaggle.deepfake.run.download", lambda config: calls.append(config.data_root)
    )

    main(["download", "--data-root", str(tmp_path)])

    assert calls == [tmp_path.resolve()]
