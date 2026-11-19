#!/bin/bash
# 一〇七平台：跑 1.1 的四个场景 + 正确性基线。
#     sbatch scripts/run_107.sbatch
#     squeue -u "$USER"                 # 看排队
#     tail -f out/slurm-<jobid>.out
#
# 配额是硬的：账号默认 qos_stu_default = cpu 4 / gpu 1 / mem 16G，
# 写 -c 8 会被拒（QOSMaxCpuPerUserLimit），显式 --qos 也会被拒。
# 所以这里只申请 4 核。
#
# 不写 --partition 之外的卡型约束：anode01-12 是 RTX 5090（sm_120），
# anode16-23 是 A100。A100 长期被占满，实际大概率落到 5090 上。
#SBATCH --job-name=nanovllm-1.1
#SBATCH --partition=Students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=out/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv107/bin/activate
mkdir -p out

echo "=== env ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'cap',torch.cuda.get_device_capability())"
python -c "import flash_attn;print('flash_attn',flash_attn.__version__)"

for s in steady_decode mixed_long_short kv_pressure many_small; do
  echo "=== $s ==="
  python -m bench.run --scenario "$s" --json "out/$s.json" 2>&1 | tee "out/$s.txt"
done

python -m bench.timeline --scenario steady_decode

echo "=== correctness: baseline ==="
python -m bench.correctness --write tests/golden/qwen3_0_6b.json || true
echo "=== correctness: check ==="
python -m bench.correctness --golden tests/golden/qwen3_0_6b.json || true

echo "=== pytest ==="
python -m pytest tests/test_1_1.py -q || true

echo "=== done $(date) ==="
