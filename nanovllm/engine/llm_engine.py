import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.stats import RequestRecord, StepRecord
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.step_records: list[StepRecord] = []
        self.request_records: list[RequestRecord] = []
        self.trace = None
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def abort_request(self, seq_ids: int | list[int]):
        self.scheduler.abort(seq_ids if isinstance(seq_ids, list) else [seq_ids])

    def step(self):
        t0 = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        t1 = perf_counter()
        finished = self.scheduler.finished
        self.scheduler.finished = []
        preempted = self.scheduler.preempted
        self.scheduler.preempted = []
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        timing = None
        if seqs:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            timing = self.model_runner.last_timing
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        t2 = perf_counter()
        # 释放放在 run 返回之后：GPU 可能还在读这些 block（2.7 的 ZOS 会放大这一点）
        for seq in finished:
            self.scheduler.block_manager.deallocate(seq)
            self.request_records.append(RequestRecord(
                seq_id=seq.seq_id,
                num_prompt_tokens=seq.num_prompt_tokens,
                num_completion_tokens=seq.num_completion_tokens,
                arrival_time=seq.arrival_time,
                finish_time=perf_counter(),
                token_times=list(seq.token_times),
                num_preemptions=seq.num_preemptions,
                finish_reason=seq.finish_reason,
            ))
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.finish_reason) for seq in finished]
        if seqs:
            cpu_runner = timing["cpu_prepare"] + timing["cpu_launch"] + timing["cpu_sample_wait"]
            record = StepRecord(
                step_id=len(self.step_records),
                is_prefill=is_prefill,
                num_seqs=len(seqs),
                num_tokens=abs(num_tokens),
                wall=t2 - t0,
                cpu_schedule=t1 - t0,
                cpu_prepare=timing["cpu_prepare"],
                cpu_launch=timing["cpu_launch"],
                cpu_sample_wait=timing["cpu_sample_wait"],
                gpu_forward=timing["gpu_forward"],
                gpu_sample=timing["gpu_sample"],
                cpu_postprocess=max(t2 - t0 - (t1 - t0) - cpu_runner, 0.),
                seq_ids=[seq.seq_id for seq in seqs],
                finished=[seq.seq_id for seq in finished],
                preempted=list(preempted),
            )
            self.step_records.append(record)
            if self.trace is not None:
                self.trace(record)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids, _ in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
