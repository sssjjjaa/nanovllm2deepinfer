#!/bin/bash
# 一〇七平台（USTC 本科生算力平台，Slurm）一次性环境准备。
#
# 在登录节点上跑（不要放进 sbatch：装包不需要 GPU，登录节点就能干）：
#     bash scripts/setup_107.sh
#
# 装完以后，跑实验用 scripts/run_107.sbatch。
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "repo: $ROOT"

# CUDA 12.6 和 torch 2.6 的 cu124 轮子兼容；集群上这两个模块都在。
module load cuda/12.6 2>/dev/null || true

PY="${PY:-python3}"
"$PY" -V

if [ ! -d .venv107 ]; then
  "$PY" -m venv .venv107
fi
source .venv107/bin/activate

PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
pip install -U pip -i "$PIP_MIRROR"

# PyPI 上 linux 的 torch 2.6.0 自带 cu124 依赖，不用去 download.pytorch.org 拉。
pip install "torch==2.6.0" -i "$PIP_MIRROR"
pip install "transformers>=4.51,<5" xxhash numpy pytest huggingface_hub -i "$PIP_MIRROR"

# flash-attn 不编源码，直接装官方预编译轮子（cp312 / cu12 / torch2.6 / cxx11abiFALSE）。
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
  -i "$PIP_MIRROR"

# 模型：0.6B 约 1.2 GB，走 hf-mirror。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
if not os.path.exists(os.path.join(path, "config.json")):
    snapshot_download("Qwen/Qwen3-0.6B", local_dir=path)
print("model:", path)
PY

python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn FAILED:", e)
PY

echo
echo "done. 提交作业： sbatch scripts/run_107.sbatch"
