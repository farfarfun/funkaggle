#!/usr/bin/env bash
# 构建 / 递增版本号 / 发布，统一走 funbuild（见组织 SPEC.md §4.4），
# 不再手写 setup.py / twine 流程，也不再在发布脚本里混入无关的 git 操作。
set -euo pipefail

command -v funbuild >/dev/null || { echo "error: funbuild is required (pip install funbuild)" >&2; exit 1; }

funbuild build "$@"
