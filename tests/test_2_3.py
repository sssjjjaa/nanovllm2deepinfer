"""2.3 验收：踢谁、踢了之后怎么办。

四条纯 CPU 测试，直接摆好 running 队列再调 `_schedule_decode`，
不经过 prefill，免得被 2.2 的尝试间隔干扰。
"""
from types import SimpleNamespace

import pytest

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams


def make_scheduler(num_blocks=6, block_size=4, max_num_seqs=8,
                   max_num_batched_tokens=1024, max_model_len=64,
                   interval=70, watermark=0.9, stop_preempted=False):
    Sequence.block_size = block_size
    cfg = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        eos=-1,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
        prefill_attempt_interval=interval,
        prefill_watermark=watermark,
        stop_preempted_request=stop_preempted,
    )
    return Scheduler(cfg)


def push_running(sched, prompt_len, out_len, tag=1):
    """直接造一个已经在 decode 的请求：先长到 target 长度，再按当前长度占块。"""
    seq = Sequence([tag] * prompt_len,
                   SamplingParams(temperature=1.0, max_tokens=out_len + 16, ignore_eos=True))
    for _ in range(out_len):
        seq.append_token(7)
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    sched.block_manager.allocate(seq, 0)
    sched.running.append(seq)
    return seq


def test_victim_is_the_longest_output():
    """放不下时踢输出最长的，不是队尾最新进来的那个。"""
    sched = make_scheduler(num_blocks=6, block_size=4)
    a = push_running(sched, prompt_len=4, out_len=9, tag=1)    # 13 tok -> 4 块
    b = push_running(sched, prompt_len=4, out_len=4, tag=2)    #  8 tok -> 2 块
    assert len(sched.block_manager.free_block_ids) == 0
    assert a.num_completion_tokens > b.num_completion_tokens

    scheduled = sched._schedule_decode()
    preempted = sched.preempted
    assert a.seq_id in preempted, "应该踢输出最长的 a"
    assert b.seq_id not in preempted
    assert a not in sched.running
    assert b in sched.running


def test_reset_for_recompute():
    """重算前把"算过多少"的记账清干净，token_ids 和 num_prompt_tokens 不动。"""
    seq = Sequence([1, 2, 3, 4], SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True))
    seq.num_cached_tokens = 4
    seq.num_scheduled_tokens = 2
    seq.total_inflight_tokens = 3
    seq.is_prefill = False
    seq.append_token(9)
    n_tokens, n_prompt = seq.num_tokens, seq.num_prompt_tokens

    seq.reset_for_recompute()

    assert seq.status == SequenceStatus.WAITING
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 0
    assert seq.total_inflight_tokens == 0
    assert seq.is_prefill is True
    assert seq.num_tokens == n_tokens, "已经吐出来的 token 不能丢"
    assert seq.num_prompt_tokens == n_prompt


def test_recompute_goes_back_to_the_head_of_waiting():
    """重算这一路：回到 waiting 队头，而且 block 当场还回去。"""
    sched = make_scheduler(num_blocks=6, block_size=4)
    a = push_running(sched, prompt_len=4, out_len=9, tag=1)
    b = push_running(sched, prompt_len=4, out_len=4, tag=2)
    free_before = len(sched.block_manager.free_block_ids)

    sched._schedule_decode()

    assert a in sched.waiting and sched.waiting[0] is a, "重算的请求要排队头，不能被新请求饿死"
    assert sched.block_manager.free_block_ids.__len__() > free_before
    assert a.block_table == [], "block 必须当场释放"
    assert sched.recomputed_tokens == 13, "白算的 token 数 = 被踢时的 num_tokens"


def test_stop_preempted_request():
    """stop 开关：被踢的直接终态，不重算，同一步就要把 block 还回来。"""
    sched = make_scheduler(num_blocks=6, block_size=4, stop_preempted=True)
    a = push_running(sched, prompt_len=4, out_len=9, tag=1)
    b = push_running(sched, prompt_len=4, out_len=4, tag=2)

    sched._schedule_decode()

    assert a.status == SequenceStatus.FINISHED_PREEMPTED
    assert a.finish_reason == "insufficient_system_resource"
    assert a in sched.finished
    assert a not in sched.waiting, "stop 这一路不能回 waiting"
    assert a.block_table == [], "同一步就要释放，否则当前请求还是放不下"
