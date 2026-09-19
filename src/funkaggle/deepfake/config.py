"""deepfake 子模块的运行时配置解析。

不依赖 tensorflow / face_recognition 等重依赖，可独立导入与测试。
路径优先级：命令行参数 > 环境变量 > 代码内默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_DATA_ROOT = "FUNKAGGLE_DATA_ROOT"

# 安全默认值：用户缓存目录下的子目录，不依赖任何人的本地绝对路径。
DEFAULT_DATA_ROOT = Path.home() / ".cache" / "funkaggle" / "deepfake"


@dataclass(frozen=True)
class DeepfakeConfig:
    """DFDC 流程用到的所有路径，均从 `data_root` 派生。

    Attributes:
        data_root: 数据集、模型权重与产出的根目录。
    """

    data_root: Path

    @property
    def train_data(self) -> Path:
        """训练集抽帧图片目录。"""
        return self.data_root / "train_data"

    @property
    def test_data(self) -> Path:
        """验证集抽帧图片目录。"""
        return self.data_root / "test_data"

    @property
    def predict_data(self) -> Path:
        """待预测抽帧图片目录。"""
        return self.data_root / "predict_data"

    @property
    def predict_source(self) -> Path:
        """待预测原始视频目录。"""
        return self.data_root / "deepfake-detection-challenge" / "test_videos"

    @property
    def submission_path(self) -> Path:
        """Kaggle 提交文件输出路径。"""
        return self.data_root / "result" / "submission.csv"


def resolve_data_root(cli_value: str | None = None) -> Path:
    """按 命令行参数 > 环境变量 > 默认值 的优先级解析数据根目录。

    Args:
        cli_value: 命令行 `--data-root` 传入的值，未传则为 None。

    Returns:
        展开 `~` 并转为绝对路径后的数据根目录。
    """
    if cli_value:
        return Path(cli_value).expanduser().resolve()

    env_value = os.environ.get(ENV_DATA_ROOT)
    if env_value:
        return Path(env_value).expanduser().resolve()

    return DEFAULT_DATA_ROOT


def build_config(cli_value: str | None = None) -> DeepfakeConfig:
    """构建 `DeepfakeConfig`。

    Args:
        cli_value: 命令行 `--data-root` 传入的值，未传则为 None。

    Returns:
        解析好路径优先级后的配置对象。
    """
    return DeepfakeConfig(data_root=resolve_data_root(cli_value))
