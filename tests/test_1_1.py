"""1.1 验收：观测工具本身的测试。

不需要 GPU 的那些用纯 Python 的假数据跑；需要 GPU 的用 skipif 标出来，
没有卡的机器上也能跑其余部分。
"""
import json
import pickle
from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.stats import RequestRecord, StepRecord
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from bench import metrics
from bench.correctness import compare, greedy_match, teacher_forced_nll
from bench.trace import TraceWriter


def make_step(**kwargs) -> StepRecord:
    base = dict(
        step_id=0, is_prefill=False, num_seqs=4, num_tokens=4, wall=0.1,
        cpu_schedule=0.001, cpu_prepare=0.002, cpu_launch=0.003, cpu_sample_wait=0.004,
        gpu_forward=0.05, gpu_sample=0.03, cpu_postprocess=0.01,
    )
    base.update(kwargs)
    return StepRecord(**base)


def make_scheduler(num_blocks=64, block_size=4, max_num_seqs=8, max_num_batched_tokens=64):
    Sequence.block_size = block_size
    cfg = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=-1,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(cfg)


def run_until_idle(sched, token_id=5, max_steps=64):
    finished, preempted = [], []
    for _ in range(max_steps):
        seqs, is_prefill = sched.schedule()
        if is_prefill:
            sched.postprocess(seqs, [token_id] * len(seqs), True)
        else:
            sched.postprocess(seqs, [token_id] * len(seqs), False)
        finished.extend(sched.finished)
        preempted.extend(sched.preempted)
        sched.finished, sched.preempted = [], []
        if sched.is_finished():
            break
    return finished, preempted


def test_gpu_busy():
    assert make_step().gpu_busy == pytest.approx(0.8)


def test_gpu_busy_zero_wall():
    assert make_step(wall=0.0).gpu_busy == 0.0


def test_step_record_segments_sum_to_wall():
    rec = make_step()
    parts = (rec.cpu_schedule + rec.cpu_prepare + rec.cpu_launch + rec.cpu_sample_wait
             + rec.gpu_forward + rec.gpu_sample + rec.cpu_postprocess)
    assert parts == pytest.approx(rec.wall)


def test_request_ttft_and_tbt():
    rec = RequestRecord(
        seq_id=0, num_prompt_tokens=10, num_completion_tokens=2, arrival_time=1.0,
        finish_time=3.0, token_times=[1.5, 2.0, 2.5], num_preemptions=0, finish_reason="length",
    )
    assert rec.ttft == pytest.approx(0.5)
    assert rec.tbt == pytest.approx([0.5, 0.5])
    assert rec.latency == pytest.approx(2.0)


def test_sequence_timestamps():
    seq = Sequence([1, 2, 3])
    assert seq.arrival_time > 0
    assert seq.token_times == []
    assert seq.num_preemptions == 0
    seq.append_token(7)
    assert len(seq.token_times) == 1


def test_sequence_getstate_excludes_observation():
    """TP > 1 时 __getstate__ 决定什么被 pickle 到别的 rank，token_times 不能进去。"""
    seq = Sequence([1, 2, 3])
    seq.append_token(7)
    state = seq.__getstate__()
    assert isinstance(state, tuple) and len(state) == 6
    revived = pickle.loads(pickle.dumps(seq))
    assert not hasattr(revived, "token_times")


def test_scheduler_finished_list():
    """finished 只在判定结束时追加；放错位置每步会把所有在跑的请求都打包一次。"""
    sched = make_scheduler()
    for _ in range(2):
        sched.add(Sequence([1, 2, 3, 4], SimpleNamespace(temperature=1.0, max_tokens=3, ignore_eos=True)))
    finished, _ = run_until_idle(sched)
    assert len(finished) == 2
    assert all(seq.is_finished for seq in finished)


def test_scheduler_preempt_records():
    sched = make_scheduler()
    seq = Sequence([1, 2, 3, 4], SimpleNamespace(temperature=1.0, max_tokens=8, ignore_eos=True))
    sched.add(seq)
    sched.schedule()
    sched.preempt(seq)
    assert seq.num_preemptions == 1
    assert sched.preempted == [seq.seq_id]
    assert seq.status == SequenceStatus.WAITING


def test_percentile():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert metrics.percentile(xs, 50) == pytest.approx(2.5)
    assert metrics.percentile(xs, 0) == pytest.approx(1.0)
    assert metrics.percentile(xs, 100) == pytest.approx(4.0)
    assert metrics.percentile([], 99) == 0.0


def test_gpu_busy_is_time_weighted():
    """30 ms 的满负荷步和 10 ms 的空转步，加权是 0.75，不是 0.5。"""
    fast = make_step(wall=0.03, gpu_forward=0.03, gpu_sample=0.0, cpu_schedule=0.0,
                     cpu_prepare=0.0, cpu_launch=0.0, cpu_sample_wait=0.0, cpu_postprocess=0.0)
    idle = make_step(wall=0.01, gpu_forward=0.0, gpu_sample=0.0, cpu_schedule=0.0,
                     cpu_prepare=0.0, cpu_launch=0.0, cpu_sample_wait=0.0, cpu_postprocess=0.0)
    summary = metrics.summarize_steps([fast, idle])
    assert summary["decode"]["gpu_busy"] == pytest.approx(0.75)


def test_summarize_requests():
    recs = [
        RequestRecord(0, 4, 2, 0.0, 1.0, [0.1, 0.3, 0.5], 0, "length"),
        RequestRecord(1, 4, 2, 0.0, 2.0, [0.2, 0.4], 1, "stop"),
    ]
    summary = metrics.summarize_requests(recs)
    assert summary["num_requests"] == 2
    assert summary["ttft"]["max"] == pytest.approx(0.2)
    assert summary["preempted_requests"] == 1
    assert summary["completion_tokens"] == 4


def test_trace_writer(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as writer:
        writer(make_step(step_id=0))
        writer(make_step(step_id=1, is_prefill=True, seq_ids=[3]))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["step_id"] for r in rows] == [0, 1]
    assert rows[1]["is_prefill"] is True
    assert rows[0]["gpu_busy"] == pytest.approx(0.8)


def test_greedy_match():
    assert greedy_match([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert greedy_match([1, 2], [1, 2, 3]) == 2
    assert greedy_match([5], [5]) == 1
    assert greedy_match([], [1]) == 0


class FakeModel:
    """logits[t] 把下一个 token 打成 20 分、其余是 0。"""

    def __init__(self, vocab=16, shift=0):
        self.vocab = vocab
        self.shift = shift

    def __call__(self, ids):
        logits = torch.zeros(len(ids), self.vocab)
        for t in range(len(ids) - 1):
            logits[t, (ids[t + 1] + self.shift) % self.vocab] = 20.0
        return logits


def test_teacher_forced_nll_alignment():
    prompt, output = [1, 2, 3], [4, 5, 6]
    perfect = teacher_forced_nll(FakeModel(), prompt, output)
    assert perfect < 1e-4
    shifted = teacher_forced_nll(FakeModel(shift=1), prompt, output)
    assert shifted > 10


def test_compare():
    golden = {"greedy_match_mean": 24.8, "nll_mean": 0.565}
    ok = {"greedy_match": [24] * 8, "nll": [0.6] * 8}
    assert compare(ok, golden) == []
    broken = {"greedy_match": [0] * 8, "nll": [1.086] * 8}
    assert len(compare(broken, golden)) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_engine_step_records():
    from nanovllm import LLM, SamplingParams

    engine = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True, max_model_len=512)
    engine.add_request([1] * 32, SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=8))
    while not engine.is_finished():
        engine.step()
    assert engine.step_records
    for rec in engine.step_records:
        assert rec.wall > 0
        assert rec.gpu_forward < rec.wall or rec.is_prefill
    assert engine.request_records
    assert engine.request_records[0].ttft > 0
    assert len(engine.step_records[0].seq_ids) == 1
