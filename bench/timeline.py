"""录时间线：只录指定的几步，导出 Chrome trace，拖进 https://ui.perfetto.dev 看。

    python -m bench.timeline --scenario steady_decode
    # 产物：out/steady_decode_prefill.json（开头 3 个 prefill 步）
    #       out/steady_decode_decode.json（第 100 步起 3 个 decode 步）
"""
import argparse
import os
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from nanovllm import SamplingParams
from bench import workloads
from bench.run import build_engine, make_prompts, warmup


def drive_with_windows(engine, specs, prompts, windows):
    """windows: [(起始步, 录几步, 输出路径)]，按起始步升序。"""
    pending = list(windows)
    idx = 0
    step = 0
    current = None
    while True:
        while idx < len(specs) and specs[idx].arrival_step <= step:
            spec = specs[idx]
            engine.add_request(prompts[idx], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=spec.output_len))
            idx += 1
        if engine.is_finished():
            if idx >= len(specs):
                break
            step += 1
            continue
        if current is None and pending and step == pending[0][0]:
            start, count, path = pending[0]
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__()
            current = {"stop": start + count, "path": Path(path), "prof": prof}
        engine.step()
        step += 1
        if current is not None and step >= current["stop"]:
            current["prof"].__exit__(None, None, None)
            current["path"].parent.mkdir(parents=True, exist_ok=True)
            current["prof"].export_chrome_trace(str(current["path"]))
            print(f"trace -> {current['path']}")
            pending.pop(0)
            current = None
    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="steady_decode")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--decode-step", type=int, default=100)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    specs = workloads.load(args.scenario, args.seed)
    prompts = make_prompts(specs, args.seed)
    engine = build_engine(args)
    warmup(engine, specs, prompts)

    out = Path(args.out)
    windows = [
        (0, args.steps, out / f"{args.scenario}_prefill.json"),
        (args.decode_step, args.steps, out / f"{args.scenario}_decode.json"),
    ]
    steps = drive_with_windows(engine, specs, prompts, windows)
    print(f"ran {steps} steps, traces in {out}")


if __name__ == "__main__":
    main()
