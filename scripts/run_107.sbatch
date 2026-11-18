#!/bin/bash
# 一〇七平台：在 1 张 A100-80GB 上跑 1.1 的四个场景 + 正确性基线。
#     sbatch scripts/run_107.sbatch
#     squeue -u "$USER"          # 看排队
#     tail -f out/slurm-<jobid>.out
#
# qos 按账号实际授权改（文档里列的是 qos_stu_default / qos_stu_small /
# qos_stu_medium / qos_stu_medium_2gpu / qos_stu_long）。
#SBATCH --job-name=nanovllm-1.1
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=out/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load cuda/12.6 2>/dev/null || true
source .venv107/bin/activate

echo "node: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
mkdir -p out

for s in steady_decode mixed_long_short kv_pressure many_small; do
  echo "=== $s ==="
  python -m bench.run --scenario "$s" --json "out/$s.json" | tee "out/$s.txt"
done

python -m bench.timeline --scenario steady_decode

echo "=== correctness: baseline ==="
python -m bench.correctness --write tests/golden/qwen3_0_6b.json
echo "=== correctness: check ==="
python -m bench.correctness --golden tests/golden/qwen3_0_6b.json

echo "=== pytest ==="
python -m pytest tests/test_1_1.py -q
