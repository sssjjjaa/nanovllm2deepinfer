"""正确性回归：拿 HuggingFace 当裁判。

    python -m bench.correctness --write tests/golden/qwen3_0_6b.json   # 存基线
    python -m bench.correctness --golden tests/golden/qwen3_0_6b.json  # 比对

两个尺子：
- greedy_match：nano 近贪心（temperature=1e-3）的输出和 HF do_sample=False
  从开头起连续相同的 token 数。位置对不上就是模型算歪了。
- nll：把 nano 的输出喂给 HF 做 teacher forcing，算平均负对数似然。
  attention 或 KV 坏了，模型会说胡话，nll 会跳。
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "In 1969, humans first landed on",
    "The quick brown fox jumps over the",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "中国历史上第一个统一的封建王朝是",
    "机器学习中，过拟合指的是模型在训练集上表现很好，但在",
    "水在标准大气压下的沸点是",
    "The three laws of thermodynamics describe how",
]
MAX_NEW_TOKENS = 32


def greedy_match(a: list[int], b: list[int]) -> int:
    """公共前缀长度，zip 到短的那个就停。"""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def teacher_forced_nll(logits_fn, prompt_ids: list[int], output_ids: list[int]) -> float:
    """给 output_ids 打分。logits_fn(input_ids) 返回 [seq_len, vocab] 的未归一化 logits。

    位置 t 的 logits 预测的是 t+1 的 token，所以给输出第一个 token 打分的是
    logits[len(prompt) - 1]。错一位算的就是另一回事。
    """
    if not output_ids:
        return 0.0
    ids = list(prompt_ids) + list(output_ids)
    logits = logits_fn(ids)
    start = len(prompt_ids) - 1
    logits = logits[start:start + len(output_ids)].float()
    logp = logits.log_softmax(dim=-1)
    target = torch.tensor(output_ids, device=logp.device).unsqueeze(-1)
    nll = -logp.gather(-1, target).squeeze(-1)
    return float(nll.mean())


def compare(actual: dict, golden: dict, greedy_ratio: float = 0.5, nll_slack: float = 0.3) -> list[str]:
    """按 8 个 prompt 的均值判卷，不按单个 prompt。

    bf16 下 logits 会平局，单个 prompt 偶尔第一个 token 就翻、greedy_match
    从 32 掉到 0 是正常噪声；真坏了是 8 个一起掉。
    """
    failures = []
    greedy = sum(actual["greedy_match"]) / len(actual["greedy_match"])
    nll = sum(actual["nll"]) / len(actual["nll"])
    if greedy < golden["greedy_match_mean"] * greedy_ratio:
        failures.append(f"greedy_match mean {greedy:.2f} < {golden['greedy_match_mean']:.2f} * {greedy_ratio}")
    if nll > golden["nll_mean"] + nll_slack:
        failures.append(f"nll mean {nll:.3f} > {golden['nll_mean']:.3f} + {nll_slack}")
    return failures


def hf_logits_fn(model, device):
    def fn(input_ids: list[int]):
        with torch.inference_mode():
            ids = torch.tensor([input_ids], device=device)
            return model(ids).logits[0]
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--write", default=None)
    parser.add_argument("--golden", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    from nanovllm import LLM, SamplingParams

    path = os.path.expanduser(args.model)
    llm = LLM(path, max_model_len=args.max_model_len, gpu_memory_utilization=0.5)
    outputs = llm.generate(PROMPTS, SamplingParams(temperature=1e-3, ignore_eos=True, max_tokens=MAX_NEW_TOKENS))
    nano_ids = [o["token_ids"] for o in outputs]

    tokenizer = AutoTokenizer.from_pretrained(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf = AutoModelForCausalLM.from_pretrained(path, dtype="auto", device_map=device).eval()
    hf_fn = hf_logits_fn(hf, device)

    greedy, nll = [], []
    for prompt, out in zip(PROMPTS, nano_ids):
        prompt_ids = tokenizer.encode(prompt)
        ids = torch.tensor([prompt_ids], device=device)
        with torch.inference_mode():
            ref = hf.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        ref_ids = ref[0, ids.shape[1]:].tolist()
        g = greedy_match(out, ref_ids)
        n = teacher_forced_nll(hf_fn, prompt_ids, out)
        greedy.append(g)
        nll.append(n)
        print(f"greedy_match={g:2d}/{MAX_NEW_TOKENS}  nll={n:.3f}  {prompt[:40]!r}")

    result = {
        "model": os.path.basename(path.rstrip("/")),
        "max_new_tokens": MAX_NEW_TOKENS,
        "greedy_match": greedy,
        "greedy_match_mean": sum(greedy) / len(greedy),
        "nll": nll,
        "nll_mean": sum(nll) / len(nll),
        "nano_token_ids": nano_ids,
    }
    print(f"greedy_match mean={result['greedy_match_mean']:.2f}  nll mean={result['nll_mean']:.3f}")

    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"golden -> {out}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.golden:
        golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        failures = compare(result, golden)
        if failures:
            for f in failures:
                print(f"FAIL: {f}")
            raise SystemExit(1)
        print("OK")


if __name__ == "__main__":
    main()
