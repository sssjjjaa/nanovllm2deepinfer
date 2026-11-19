import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # 2.2：prefill 不再每步都做。每 prefill_attempt_interval 个 decode 步
    # 才尝试一次，且 KV 用量要在这条水位线以下（按 block 数算，不按 token）。
    prefill_attempt_interval: int = 70
    prefill_watermark: float = 0.9
    # 2.3：被抢占的请求是重算（False）还是直接结束（True）。
    # 有 prefill 通道的引擎可以重算；没有 prefill 通道的 decode 机器必须开。
    stop_preempted_request: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
