#!/bin/bash
# 一〇七平台（USTC 本科生算力平台，Slurm）一次性环境准备。
#
# 在登录节点上跑，不要放进 sbatch —— 装包不需要 GPU：
#     bash scripts/setup_107.sh
#
# 两个和课程文档不一样的地方，都是被平台逼出来的：
#   1. 系统 python3 没装 venv/ensurepip，`python3 -m venv` 会报
#      "ensurepip is not available"。这里用 uv 代替（自带 venv + 依赖解析）。
#   2. Students 分区能拿到的卡是 RTX 5090（sm_120），torch 2.6 没有对应
#      kernel、flash-attn 2.7.4 也不支持。这里换 torch 2.8.0 + flash-attn 2.8.3。
#      代价：数字不能和文档里的 L4 直接对比，只能比趋势。
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "repo: $ROOT"

MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# 集群直连 github 极慢（clone 会卡住），走公共加速前缀。
GH_PROXY="${GH_PROXY:-https://ghfast.top}"
VENV="${VENV:-.venv107}"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  python3 -m pip install --user -q --break-system-packages uv -i "$MIRROR"
fi
uv --version

uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" \
  "torch==2.8.0" "transformers>=4.51,<5" xxhash numpy pytest huggingface_hub \
  --index-url "$MIRROR"

# RTX 5090 = sm_120。flash-attn 2.8.3 才开始带 Blackwell 的 kernel，
# 且必须匹配 torch 2.8 + cxx11abi=TRUE（PyPI 的 torch 是 TRUE）。
WHL="flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
WHL_PATH="${TMPDIR:-/tmp}/$WHL"
if [ ! -f "$WHL_PATH" ]; then
  echo "downloading $WHL"
  curl -sL --max-time 900 -o "$WHL_PATH" \
    "$GH_PROXY/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$WHL"
fi
uv pip install --python "$VENV/bin/python" "$WHL_PATH"

# 模型：0.6B 约 1.2 GB，走 hf-mirror。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
"$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
if not os.path.exists(os.path.join(path, "config.json")):
    snapshot_download("Qwen/Qwen3-0.6B", local_dir=path)
print("model:", path)
PY

"$VENV/bin/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn FAILED:", e)
PY

echo
echo "done. 提交作业： sbatch scripts/run_107.sbatch"
