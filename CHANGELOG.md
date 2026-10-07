# CHANGELOG

## [Unreleased]

### 修复

- 下载流程改用 Kaggle 官方客户端执行认证和比赛文件下载，不再将未认证的 URL 交给通用 HTTP 下载器。
- 移除未被运行时代码使用的包内训练权重；训练产生的权重始终位于用户数据根目录。

### 变更

- README 明确 Kaggle 凭据、接受比赛规则的前置条件，以及权重文件的实际位置。

## [0.0.3]

### 新增

- 新增 `funkaggle/deepfake/config.py`，统一解析数据根目录（命令行参数 > 环境变量 `FUNKAGGLE_DATA_ROOT` > 默认值），替代原先写死的作者本地 Mac 路径。
- 新增 `tests/test_config.py`、`tests/test_download.py`、`tests/test_run_cli.py`、`tests/test_feature_model.py`，覆盖配置解析、下载模块与 CLI 的正常路径与边界（越界 index 等）。
- `run.py` 补充为标准 argparse CLI（`download` / `feature` / `train` / `predict` 四个子命令）。
- `tests/test_model.py` 覆盖 `MyModel.build/load/train/clear/predict` 全部公开 API（均用轻量测试替身，不依赖真实 tensorflow 训练）。
- `download.py` 的 `unzip_file()` 新增 zip 成员路径校验，拒绝绝对路径、`..` 等试图路径穿越目标目录的成员。

### 修复

- 修复 `run.py` 模块级直接执行 `model_predict()` 的问题：原代码只要 `import funkaggle.deepfake.run` 就会立即触发一次完整推理流程，现改为仅在 `python -m funkaggle.deepfake.run <action>` 或显式调用时才执行。
- 修复 `MyModel._init()` 用 `os.mkdir` 在父目录不存在时会直接报错的问题，改为 `os.makedirs(..., exist_ok=True)`。
- 修复 `download.py` 从与本组织同名但不相关的第三方 PyPI 包 `funtool`（作者 Active Learning Lab，与 farfarfun 无关）导入 `download` 函数的问题，改为组织自有的 `funget`。
- 删除源码中硬编码的 50 条带 `GoogleAccessId`/`Expires`/`Signature` 的 GCS 签名下载直链，改用不含凭据的 Kaggle 官方下载地址。
- `download_file()` 默认保存目录从写死的 `/root/dataset/deepfake` 改为用户缓存目录 `~/.cache/funkaggle/deepfake`。
- `download_file()` 不再忽略 `funget.download()` 的返回值：下载失败时抛出带 URL/保存路径/分卷编号的 `RuntimeError`，不再继续对失败产物解压。
- `unzip_file()` 解压前校验文件存在且是合法 zip，非法归档抛出带文件路径上下文的 `RuntimeError`/`FileNotFoundError`，不再对损坏或缺失的下载产物静默解压。

### 变更

- **Breaking**：`requires-python` 从 `>=3.10` 提到 `>=3.12`，因新引入的 `funget>=1.1.63` 硬性要求 Python 3.12+。
  迁移方法：本地与 CI 均需升级到 Python 3.12+ 再安装本包；本仓库未发布到 PyPI，无外部使用者受影响。
- **Breaking**：源码从仓库根目录 `funkaggle/` 迁移到标准 `src/funkaggle/` 布局，`pyproject.toml` 打包配置同步更新。
  迁移方法：仅影响直接引用仓库根目录路径（而非 `pip install`/`import funkaggle`）的旧脚本，需把路径引用改成 `src/funkaggle/...`；通过 `pip install -e .` 或正式安装使用本包的调用方无需改动。
- 依赖管理：将 `tensorflow`、`face-recognition`、`pandas`、`numpy`、`tqdm`、`demjson3`、`farlog`、`funget` 补充进 `pyproject.toml` 的 `[project].dependencies` 并全部加上版本下限（此前只声明了裸版本的 `kaggle`、`opencv-python`，其余运行时依赖完全未声明）。
- 依赖替换：`demjson`（已不维护，Python 2 时代的包）替换为社区维护的 Python 3 分支 `demjson3`。
- 依赖新增：引入组织自有的 `funfile>=1.0.39` 处理 zip 解压，替代直接使用标准库 `zipfile`。
- 构建发布：`script/build.sh` 不再手写 `setup.py build/sdist/bdist_egg/bdist_wheel` + `twine upload`（仓库里根本没有 `setup.py`，这段流程此前是失效的），改为调用 `funbuild build`；发布脚本不再混入 `git pull/add/commit/push`。删除了单独的 `script/push.sh`（内容与旧 `build.sh` 里的自动提交重复，且 `commit -m "add"` 不符合组织提交信息规范）。
- 日志：`model.py` 里的 `print(predict_generator.filenames)` / `print(df3)` 改为 `farlog` 的 `logger.info(...)`。
- 文档：README 补充安装命令、最小可运行示例，末尾追加组织统一的「关于 farfarfun」区块；修正源码路径引用为 `src/funkaggle/...`；说明 CLI `feature` 子命令目前只对待预测视频抽帧，训练集/验证集抽帧需直接调用 Python API；不再声称下载直链「已过期」，改为准确描述为依赖 Kaggle 官方认证。
- （历史记录）包名与导入名从 `notekaggle` 改为 `funkaggle`，与仓库名对齐。

### 废弃

- 无。经核实 `pypi.org/pypi/notekaggle` 与 `pypi.org/pypi/funkaggle` 均返回 404——两者从未发布到 PyPI，因此不存在需要转发的旧包，SPEC §15.2 的旧包转发流程不适用。
