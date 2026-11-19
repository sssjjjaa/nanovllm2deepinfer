"""一条命令跑完一个场景：报告 + 可选 JSONL trace + 可选 JSON 汇总。

本机没有 GPU 时：
    modal run scripts/modal_run.py --cmd "python -m bench.run --scenario steady_decode"
产物写在 out/，跑完用 modal volume get 拉回来。
"""
import argparse
import json
import os
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from bench import workloads
from bench.metrics import format_report, summarize_requests, summarize_steps
from bench.trace import TraceWriter


def build_engine(args):
    return LLM(
        os.path.expanduser(args.model),
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        stop_preempted_request=args.stop_preempted_request,
    )


def make_prompts(specs, seed):
    rng = Random(seed + 1)
    return [workloads.prompt_token_ids(spec, rng) for spec in specs]


def warmup(engine, specs, prompts):
    """覆盖真实负载会出现的形状，包括 batch=1。

    torch.compile 遇到没见过的形状会当场重编译 1 秒多，混进测量；
    对大小 1 的维度还要单独特化一次，所以单独再跑一个单请求。
    """
    for spec, prompt in list(zip(specs, prompts))[:64]:
        engine.add_request(prompt, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=min(spec.output_len, 64)))
    while not engine.is_finished():
        engine.step()
    engine.add_request(prompts[0][:8], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=4))
    while not engine.is_finished():
        engine.step()
    engine.step_records.clear()
    engine.request_records.clear()


def drive(engine, specs, prompts, num_steps=None, trace=None):
    """按 arrival_step 塞请求，引擎空闲但还有请求没到时只推进步数。"""
    engine.trace = trace
    idx = 0
    step = 0
    t_start = perf_counter()
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
        torch.cuda.nvtx.range_push(f"step {step}")
        engine.step()
        torch.cuda.nvtx.range_pop()
        step += 1
        if num_steps is not None and step >= num_steps:
            break
    elapsed = perf_counter() - t_start
    engine.trace = None
    return elapsed, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="steady_decode")
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--trace", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--stop-preempted-request", action="store_true",
                        help="被抢占的请求直接结束，不重算（2.3）")
    args = parser.parse_args()

    specs = workloads.load(args.scenario, args.seed)
    prompts = make_prompts(specs, args.seed)
    engine = build_engine(args)
    warmup(engine, specs, prompts)

    writer = TraceWriter(args.trace) if args.trace else None
    torch.cuda.profiler.start()
    elapsed, num_engine_steps = drive(engine, specs, prompts, args.num_steps, writer)
    torch.cuda.profiler.stop()
    if writer is not None:
        writer.close()

    total_tokens = sum(r.num_completion_tokens for r in engine.request_records)
    summary = {
        "scenario": args.scenario,
        "elapsed": elapsed,
        "engine_steps": num_engine_steps,
        "total_tokens": total_tokens,
        "throughput": total_tokens / elapsed if elapsed > 0 else 0.0,
        "steps": summarize_steps(engine.step_records),
        "requests": summarize_requests(engine.request_records),
    }
    print(f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, Throughput: {summary['throughput']:.2f}tok/s")
    print(format_report(summary))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json -> {path}")


if __name__ == "__main__":
    main()
