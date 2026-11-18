from dataclasses import dataclass, field


@dataclass(slots=True)
class StepRecord:
    """引擎一步的耗时拆分。所有时间单位都是秒。

    7 段：cpu_schedule / cpu_prepare / cpu_launch / cpu_sample_wait /
    gpu_forward / gpu_sample / cpu_postprocess，加起来约等于 wall。
    gpu_forward 和 gpu_sample 由 CUDA event 量出，其余是 perf_counter。
    """
    step_id: int
    is_prefill: bool
    num_seqs: int
    num_tokens: int
    wall: float
    cpu_schedule: float
    cpu_prepare: float
    cpu_launch: float
    cpu_sample_wait: float
    gpu_forward: float
    gpu_sample: float
    cpu_postprocess: float
    seq_ids: list[int] = field(default_factory=list)
    finished: list[int] = field(default_factory=list)
    preempted: list[int] = field(default_factory=list)

    @property
    def gpu_busy(self):
        return (self.gpu_forward + self.gpu_sample) / self.wall if self.wall > 0 else 0.0


@dataclass(slots=True)
class RequestRecord:
    """一个请求从进来到结束的全部时间戳。"""
    seq_id: int
    num_prompt_tokens: int
    num_completion_tokens: int
    arrival_time: float
    finish_time: float
    token_times: list[float]
    num_preemptions: int
    finish_reason: str

    @property
    def ttft(self):
        return self.token_times[0] - self.arrival_time if self.token_times else 0.0

    @property
    def tbt(self):
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]

    @property
    def latency(self):
        return self.finish_time - self.arrival_time
