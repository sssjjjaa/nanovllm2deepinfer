"""2.1 验收：状态机、三态资源判断、abort、没有泄漏。

全部是纯 CPU 测试：用假的 Config 直接测 Scheduler，用一个
engine_step() 复刻 LLMEngine.step() 的三段（调度 → 判定 → run 返回后释放）。
"""
from types import SimpleNamespace

import pytest

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams


def make_scheduler(num_blocks=16, block_size=4, max_num_seqs=8,
                   max_num_batched_tokens=1024, max_model_len=64):
    Sequence.block_size = block_size
    cfg = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        eos=-1,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(cfg)


def engine_step(sched, token_id=5):
    """复刻 LLMEngine.step()：schedule → postprocess → run 返回之后释放。"""
    seqs, is_prefill = sched.schedule()
    finished = sched.finished
    sched.finished = []
    if seqs:
        sched.postprocess(seqs, [token_id] * len(seqs), is_prefill)
    for seq in finished:
        sched.block_manager.deallocate(seq)
    return seqs, finished


def run_until_idle(sched, max_steps=500):
    finished = []
    for _ in range(max_steps):
        seqs, done = engine_step(sched)
        finished.extend(done)
        if sched.is_finished() and not seqs:
            return finished
    raise AssertionError("engine did not become idle")


def assert_no_leak(sched):
    bm = sched.block_manager
    free = set(bm.free_block_ids)
    assert len(bm.used_block_ids & free) == 0
    assert len(bm.used_block_ids | free) == len(bm.blocks)
    return len(free)


def seq_of(prompt_len, max_tokens=4, tag=1):
    """tag 用来区分 prompt 内容：内容一样会被 prefix cache 命中，两个请求共块，
    测"暂时放不下"就测不出来了。"""
    return Sequence([tag] * prompt_len, SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True))


def test_oversized_prompt_is_ignored():
    """验收 1：比总容量还大的 prompt 被拒绝，后面的请求照常跑完。"""
    sched = make_scheduler(num_blocks=8, block_size=4)
    short_a = seq_of(4, max_tokens=2, tag=1)
    huge = seq_of(40, max_tokens=2, tag=2)          # 10 块 >= 8 块 → NEVER
    short_b = seq_of(4, max_tokens=2, tag=3)
    for seq in (short_a, huge, short_b):
        sched.add(seq)
    finished = run_until_idle(sched)
    assert huge.status == SequenceStatus.FINISHED_IGNORED
    assert huge.finish_reason == "prompt_length_exceeded"
    assert short_a.is_finished and short_b.is_finished
    assert {s.seq_id for s in finished} == {short_a.seq_id, huge.seq_id, short_b.seq_id}
    assert assert_no_leak(sched) == len(sched.block_manager.blocks)


def test_later_waits_then_runs():
    """验收 2：暂时放不下的等前面的结束，最后正常收场。"""
    sched = make_scheduler(num_blocks=3, block_size=4)
    a = seq_of(8, max_tokens=2, tag=1)       # 2 块
    b = seq_of(8, max_tokens=2, tag=2)       # 2 块，暂时只剩 1 块 → LATER
    sched.add(a)
    sched.add(b)
    seqs, _ = engine_step(sched)
    assert [s.seq_id for s in seqs] == [a.seq_id]
    assert b.status == SequenceStatus.WAITING
    run_until_idle(sched)
    assert a.is_finished and b.is_finished
    assert b.finish_reason in ("stop", "length")
    assert_no_leak(sched)


def test_abort_running_releases_blocks():
    """验收 3：abort running 的请求，下一步不再被调度，block 全部回来。"""
    sched = make_scheduler(num_blocks=16, block_size=4, max_model_len=64)
    a = seq_of(8, max_tokens=16, tag=1)
    b = seq_of(8, max_tokens=16, tag=2)
    sched.add(a)
    sched.add(b)
    engine_step(sched)
    assert not a.block_table == []
    sched.abort([a.seq_id])
    seqs, finished = engine_step(sched)
    scheduled_ids = [s.seq_id for s in seqs]
    assert a.seq_id not in scheduled_ids
    assert [s.seq_id for s in finished] == [a.seq_id]
    assert a.finish_reason == "abort"
    assert a.block_table == []
    assert_no_leak(sched)
    run_until_idle(sched)
    assert assert_no_leak(sched) == len(sched.block_manager.blocks)


def test_abort_waiting_releases_nothing_but_leaves_queue():
    """验收 4：abort waiting 的请求，没有 block 要还，但不能留在队列里。"""
    sched = make_scheduler(num_blocks=3, block_size=4)
    a = seq_of(8, max_tokens=2, tag=1)
    b = seq_of(8, max_tokens=2, tag=2)
    sched.add(a)
    sched.add(b)
    engine_step(sched)
    assert b.status == SequenceStatus.WAITING
    sched.abort([b.seq_id])
    seqs, finished = engine_step(sched)
    assert [s.seq_id for s in finished] == [b.seq_id]
    assert b.finish_reason == "abort"
    run_until_idle(sched)
    assert_no_leak(sched)


def test_all_terminal_states_go_through_one_exit():
    """验收 5：正常结束、截断、取消、拒绝四种结局都能出队，结束时没有泄漏。"""
    sched = make_scheduler(num_blocks=12, block_size=4)
    stop = seq_of(4, max_tokens=2, tag=1)
    capped = seq_of(4, max_tokens=1, tag=2)
    aborted = seq_of(4, max_tokens=8, tag=3)
    for seq in (stop, capped, aborted):
        sched.add(seq)
    engine_step(sched)
    sched.abort([aborted.seq_id])
    finished = run_until_idle(sched)
    assert {s.seq_id for s in finished} == {stop.seq_id, capped.seq_id, aborted.seq_id}
    assert capped.finish_reason == "length"
    assert aborted.finish_reason == "abort"
    assert assert_no_leak(sched) == len(sched.block_manager.blocks)


def test_prompt_at_model_len_minus_one():
    """验收 6：prompt 长度 max_model_len - 1，max_tokens=1 能正常产出 1 个 token。"""
    sched = make_scheduler(num_blocks=16, block_size=4, max_model_len=32)
    seq = seq_of(31, max_tokens=1)
    sched.add(seq)
    finished = run_until_idle(sched)
    assert finished == [seq]
    assert seq.num_completion_tokens == 1
    assert seq.num_tokens == 32
    assert seq.finish_reason == "length"
    assert_no_leak(sched)


def test_prompt_over_model_len_is_rejected_at_add():
    """入队就拒绝，而不是等到 decode 越界。"""
    sched = make_scheduler(num_blocks=64, block_size=4, max_model_len=32)
    seq = seq_of(32, max_tokens=4)
    sched.add(seq)
    assert seq.status == SequenceStatus.FINISHED_IGNORED
    finished = run_until_idle(sched)
    assert finished == [seq]
    assert seq.finish_reason == "prompt_length_exceeded"
