"""四个场景的请求列表。纯 Python，不 import torch。

每个请求写成 RequestSpec：prompt 长度、输出长度、在第几步到达。
到达时间按"步数"定义，不按墙钟——GPU 快慢抖动不会改变调度决策，
同一个场景在 L4 和 H100 上跑出来的调度序列是一样的，才能前后对比。
"""
from dataclasses import dataclass
from random import Random


@dataclass(slots=True)
class RequestSpec:
    prompt_len: int
    output_len: int
    arrival_step: int = 0


def steady_decode(seed: int = 0) -> list[RequestSpec]:
    """原 bench.py 的负载：256 个请求一起到，prompt / 输出都在 100-1024。"""
    rng = Random(seed)
    return [RequestSpec(rng.randint(100, 1024), rng.randint(100, 1024)) for _ in range(256)]


def mixed_long_short(seed: int = 0) -> list[RequestSpec]:
    """128 个短请求稳态 decode，第 100 步起每 20 步插一个 3072 token 的长 prompt。"""
    rng = Random(seed)
    specs = [RequestSpec(rng.randint(100, 200), 256) for _ in range(128)]
    for step in range(100, 100 + 20 * 8, 20):
        specs.append(RequestSpec(3072, 128, step))
    return specs


def kv_pressure(seed: int = 0) -> list[RequestSpec]:
    """512 个请求，输出 768-1024：显存一定会被 KV 吃爆，用来看抢占。"""
    rng = Random(seed)
    return [RequestSpec(rng.randint(100, 1024), rng.randint(768, 1024)) for _ in range(512)]


def many_small(seed: int = 0) -> list[RequestSpec]:
    """512 个 16-32 in / 32-64 out：GPU 时间短，CPU 固定开销显出来。"""
    rng = Random(seed)
    return [RequestSpec(rng.randint(16, 32), rng.randint(32, 64)) for _ in range(512)]


SCENARIOS = {
    "steady_decode": steady_decode,
    "mixed_long_short": mixed_long_short,
    "kv_pressure": kv_pressure,
    "many_small": many_small,
}


def load(name: str, seed: int = 0) -> list[RequestSpec]:
    if name not in SCENARIOS:
        raise SystemExit(f"unknown scenario: {name}, choose from {list(SCENARIOS)}")
    return SCENARIOS[name](seed)


def prompt_token_ids(spec: RequestSpec, rng: Random) -> list[int]:
    """随机 token id 当 prompt，和原 bench.py 一致，省掉 tokenizer 的开销。"""
    return [rng.randint(0, 10000) for _ in range(spec.prompt_len)]
