# funkaggle

Kaggle Deepfake Detection Challenge（DFDC，2019-2020）的一次性参赛脚本，目前**未维护**，仅作代码留存。

> 注意：仓库名是 `funkaggle`，但 Python 包名、导入名（以及若发布到 PyPI 时的名字）都是 `notekaggle`——这是旧的 `note*` → `fun*` 改名遗留问题，改名时漏掉了这个包，目前尚未发布到 PyPI。

## 代码结构

所有逻辑都在 `notekaggle.deepfake` 子模块里，围绕 DFDC 比赛的下载 / 抽帧 / 训练 / 预测四步：

- `notekaggle/deepfake/download.py`：`download_files()` / `download_file()` 按编号下载 `dfdc_train_part_XX.zip` 训练数据分卷并解压。里面的下载地址是当年 Kaggle 生成的带签名 GCS 直链，**早已过期，无法直接使用**。
- `notekaggle/deepfake/feature.py`：`video2img_train()` / `video2img_predict()` 用 `face_recognition` 从训练/测试视频里抽帧、裁出人脸区域，按真假标签分目录保存为图片。
- `notekaggle/deepfake/model.py`：`MyModel` 类，一个基于 `tf.keras` 的简单三层卷积二分类模型，提供 `build()` / `train()` / `load()` / `predict()`，`predict()` 会直接生成 Kaggle 提交用的 `submission.csv`。
- `notekaggle/deepfake/run.py`：把上面几步串起来的入口脚本，`download()` / `feature()` / `model_train()` / `model_predict()`；其中路径是写死的作者本地 Mac 路径（如 `/Users/liangtaoniu/tmp/dataset/deepfake/`），且模块级直接执行了 `model_predict()`，不能直接当库导入使用。

仓库里还附带了一份训练好的权重 `notekaggle/models/deepfake/weights.hdf5`。

## 安装

未发布到 PyPI（`notekaggle` / `funkaggle` 均查询不到），如需使用需克隆本仓库后本地安装依赖（`kaggle`、`opencv-python`，以及代码里用到但未在 `pyproject.toml` 声明的 `tensorflow`、`face_recognition`、`demjson`、`pandas` 等）。

## 使用

由于路径写死、依赖较老（`tensorflow` 1.x/2.x 风格 API、`demjson`、`face_recognition` 依赖 dlib 编译）且下载链接已过期，本项目不具备开箱即用的能力，主要作为当年参赛代码的存档，如需复用建议直接阅读 `notekaggle/deepfake/` 下的源码按需改写。
