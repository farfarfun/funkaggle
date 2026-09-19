"""DFDC deepfake 比赛流程 CLI 入口：下载 / 抽帧 / 训练 / 预测。

数据根目录按 命令行参数 `--data-root` > 环境变量 `FUNKAGGLE_DATA_ROOT` > 默认值
（`~/.cache/funkaggle/deepfake`）的优先级解析，不写死任何人的本地路径。

用法::

    python -m funkaggle.deepfake.run download --data-root /data/deepfake
    python -m funkaggle.deepfake.run feature
    python -m funkaggle.deepfake.run train
    python -m funkaggle.deepfake.run predict

下载步骤依赖的直链已过期（见 `download.py`），`feature` / `train` / `predict`
需要自行准备好对应目录下的视频或图片数据。
"""

from __future__ import annotations

import argparse

from farlog import getLogger

from funkaggle.deepfake.config import DeepfakeConfig, build_config

logger = getLogger("funkaggle")


def download(config: DeepfakeConfig, file_index_list: list[int] | None = None) -> None:
    """下载 DFDC 训练数据分卷（下载直链已过期，仅作流程演示）。

    Args:
        config: 运行时配置。
        file_index_list: 待下载的分卷编号列表，默认前 5 个分卷。
    """
    from funkaggle.deepfake.download import download_files

    file_index_list = list(range(5)) if file_index_list is None else file_index_list
    download_files(save_dir=str(config.data_root), file_index_list=file_index_list)


def feature(
    config: DeepfakeConfig,
    test: bool = False,
    train: bool = False,
    predict: bool = False,
) -> None:
    """从视频中抽取人脸帧，按训练/测试/预测拆分保存。

    Args:
        config: 运行时配置。
        test: 是否为验证集（`dfdc_train_part_0`）抽帧。
        train: 是否为训练集（`dfdc_train_part_2/3`）抽帧。
        predict: 是否为待预测视频抽帧。
    """
    from funkaggle.deepfake.feature import video2img_predict, video2img_train

    if test:
        for index in [0]:
            path = config.data_root / f"dfdc_train_part_{index}"
            video2img_train(str(path), target_dir=str(config.test_data))
    if train:
        for index in [2, 3]:
            path = config.data_root / f"dfdc_train_part_{index}"
            video2img_train(str(path), target_dir=str(config.train_data))
    if predict:
        video2img_predict(
            str(config.predict_source), target_dir=str(config.predict_data)
        )


def model_train(config: DeepfakeConfig) -> None:
    """基于 `config` 指定的目录训练二分类模型。

    Args:
        config: 运行时配置。
    """
    from funkaggle.deepfake.model import MyModel

    model = MyModel(
        data_root=str(config.data_root),
        train_data_dir=str(config.train_data),
        test_data_dir=str(config.test_data),
    )
    model.build()
    model.load()
    model.train(batch_size=32)


def model_predict(config: DeepfakeConfig) -> None:
    """加载已训练模型，对 `predict_data` 生成 Kaggle 提交文件。

    Args:
        config: 运行时配置。
    """
    from funkaggle.deepfake.model import MyModel

    model = MyModel(
        data_root=str(config.data_root),
        train_data_dir=str(config.train_data),
        test_data_dir=str(config.test_data),
    )
    model.build()
    model.load()
    model.predict(
        predict_dir=str(config.predict_data),
        submission_path=str(config.submission_path),
    )


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(prog="funkaggle-deepfake", description=__doc__)
    parser.add_argument(
        "action",
        choices=["download", "feature", "train", "predict"],
        help="要执行的步骤",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="数据集根目录；未传时读取环境变量 FUNKAGGLE_DATA_ROOT，都没有则使用默认缓存目录",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI 入口。"""
    args = _build_parser().parse_args(argv)
    config = build_config(args.data_root)
    logger.info(f"data_root={config.data_root}")

    if args.action == "download":
        download(config)
    elif args.action == "feature":
        feature(config, predict=True)
    elif args.action == "train":
        model_train(config)
    elif args.action == "predict":
        model_predict(config)


if __name__ == "__main__":
    main()
