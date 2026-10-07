# funkaggle

Kaggle Deepfake Detection Challenge（DFDC，2019-2020）的一次性参赛脚本：下载数据 / 从视频抽人脸帧 / 训练一个简单三层卷积二分类模型 / 生成提交文件。目前**未维护**，仅作代码留存；下载步骤依赖 Kaggle 官方认证（需配置好 Kaggle API 凭据并接受比赛规则），未配置时会下载失败，需要自备数据才能真正跑起来。

## 安装

尚未发布到 PyPI，需本地克隆后安装：

```bash
git clone https://github.com/farfarfun/funkaggle.git
cd funkaggle
uv pip install -e .
# 或
pip install -e .
```

`requires-python = ">=3.12"`。

## 最小可运行示例

不依赖真实数据集，验证安装与 CLI 是否正常：

```bash
python -m funkaggle.deepfake.run --help
```

配置解析（命令行参数 > 环境变量 `FUNKAGGLE_DATA_ROOT` > 默认值）可直接在 Python 里调用：

```python
from funkaggle.deepfake.config import build_config

config = build_config("/data/deepfake")  # 或不传，走环境变量/默认值
print(config.train_data)  # /data/deepfake/train_data
print(config.submission_path)  # /data/deepfake/result/submission.csv
```

真正跑完整流程前，下载步骤需要在 Kaggle 中接受 DFDC 比赛规则，并配置
`~/.kaggle/kaggle.json` 或环境变量 `KAGGLE_USERNAME` / `KAGGLE_KEY`。代码通过
Kaggle 官方客户端认证和下载；其余步骤则需要已下载或自行准备的数据：

```bash
python -m funkaggle.deepfake.run download --data-root /data/deepfake
python -m funkaggle.deepfake.run feature   --data-root /data/deepfake  # 仅对待预测视频抽帧
python -m funkaggle.deepfake.run train     --data-root /data/deepfake
python -m funkaggle.deepfake.run predict   --data-root /data/deepfake
```

CLI 的 `feature` 子命令目前只对待预测视频抽帧（等价于 `feature(config, predict=True)`），
没有暴露训练集/验证集抽帧的开关。`train` 子命令需要的训练集/验证集图片要先通过
Python API 单独生成：

```python
from funkaggle.deepfake.run import build_config, feature

config = build_config("/data/deepfake")
feature(config, train=True, test=True)  # 对训练集/验证集视频抽帧，供 train 子命令使用
```

## 代码结构

所有逻辑都在 `src/funkaggle/deepfake` 子模块里，围绕 DFDC 比赛的下载 / 抽帧 / 训练 / 预测四步：

- `src/funkaggle/deepfake/config.py`：路径配置解析，命令行参数 > 环境变量 `FUNKAGGLE_DATA_ROOT` > 默认值，不写死任何人的本地路径。
- `src/funkaggle/deepfake/download.py`：通过 Kaggle 官方客户端认证后按编号下载 `dfdc_train_part_XX.zip` 训练数据分卷并安全解压（校验成员路径，拒绝越出目标目录）。
- `src/funkaggle/deepfake/feature.py`：`video2img_train()` / `video2img_predict()` 用 `face_recognition` 从训练/测试视频里抽帧、裁出人脸区域，按真假标签分目录保存为图片。
- `src/funkaggle/deepfake/model.py`：`MyModel` 类，一个基于 `tf.keras` 的简单三层卷积二分类模型，提供 `build()` / `train()` / `load()` / `predict()`，`predict()` 会生成 Kaggle 提交用的 `submission.csv`。
- `src/funkaggle/deepfake/run.py`：CLI 入口，把上面几步串起来（`download` / `feature` / `train` / `predict` 四个子命令）。

训练得到的权重保存在数据根目录的 `models/weights.hdf5`，不随源码包发布。

## 局限性

由于下载依赖 Kaggle 官方认证、依赖较老（`tensorflow`、`face_recognition` 依赖 dlib 编译）且需自行准备数据，本项目不具备开箱即用的能力，主要作为当年参赛代码的存档；如需复用建议直接阅读 `src/funkaggle/deepfake/` 下的源码按需改写。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
