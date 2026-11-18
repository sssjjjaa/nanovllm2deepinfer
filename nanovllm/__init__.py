__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # 惰性导入：`nanovllm.engine.scheduler`、`stats` 这些纯 Python 模块
    # 不该因为 LLM 会拉 triton / flash-attn 就连带装不上。
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
