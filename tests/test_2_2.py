"""2.2 验收：prefill 不再每步都做。

四条纯 CPU 测试，用假 Config 直接测 Scheduler：
  (a) 不在尝试点时，waiting 非空也走 decode
  (b) KV 用量在水位线之上不 prefill
  (c) prefill 排不出时转 decode，不返回空步
  (e) prefill 一旦开始就连着做（decode_step 清零）
  (f) 放不下时切一刀，被切的排在 batch 末尾
2.1 的测试（tests/test_2_1.py）继续跑，就是 (d)。
"""
from types import SimpleNamespace

import pytest

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams


def make_scheduler(num_blocks=16, block_size=4, max_num_seqs=8,
                   max_num_batched_tokens=1024, max_model_len=64,
                   interval=70, watermark=0.9):
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
    )
    return Scheduler(cfg)


def seq_of(prompt_len, max_tokens=4, tag=1):
    """tag 区分 prompt 内容：内容相同会被 prefix cache 命中，两个请求共块。"""
    return Sequence([tag] * prompt_len,
                    SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True))


def engine_step(sched, token_id=5):
    """复刻 LLMEngine.step()：schedule → postprocess → run 返回之后释放。"""
    seqs, is_prefill = sched.schedule()
    finished = sched.finished
    sched.finished = []
    if seqs:
        sched.postprocess(seqs, [token_id] * len(seqs), is_prefill)
    for seq in finished:
        sched.block_manager.deallocate(seq)
    return seqs, is_prefill


def test_a_not_attempt_point_goes_decode():
    """(a) 默认 interval=70：decode 了 1 步之后新请求进来，不能抢 prefill。"""
    sched = make_scheduler()
    a = seq_of(8, max_tokens=32, tag=1)
    sched.add(a)
    seqs, is_prefill = engine_step(sched)
    assert is_prefill and a in seqs
    assert sched.decode_step == 0          # prefill 步把 decode_step 清零

    seqs, is_prefill = engine_step(sched)   # waiting 空 → decode
    assert not is_prefill
    assert sched.decode_step == 1

    b = seq_of(8, max_tokens=4, tag=2)
    sched.add(b)
    seqs, is_prefill = sched.schedule()     # 1 % 70 != 0 → 不尝试 prefill
    assert not is_prefill, "不在尝试点，waiting 非空也必须走 decode"
    assert b not in seqs
    assert a in seqs


def test_b_watermark_blocks_prefill():
    """(b) KV 用量高于水位线时不 prefill，先让 decode 跑。"""
    sched = make_scheduler(num_blocks=16, watermark=0.0)
    a = seq_of(40, max_tokens=8, tag=1)
    sched.add(a)
    seqs, is_prefill = engine_step(sched)   # running 空 → 空闲例外，允许 prefill
    assert is_prefill
    assert sched.kv_usage > 0.0

    b = seq_of(8, max_tokens=4, tag=2)
    sched.add(b)
    seqs, is_prefill = sched.schedule()
    assert sched.kv_usage > sched.prefill_watermark
    assert not is_prefill, "水位线之上不能 prefill"
    assert b not in seqs


def test_c_prefill_no_room_falls_back_to_decode():
    """(c) waiting 队头放不下（LATER）时转 decode，不返回空步。"""
    sched = make_scheduler(num_blocks=4, block_size=4)
    a = seq_of(12, max_tokens=4, tag=1)     # 3 块
    sched.add(a)
    seqs, is_prefill = engine_step(sched)
    assert is_prefill and a in seqs

    b = seq_of(8, max_tokens=4, tag=2)      # 要 2 块，只剩 1 块 → LATER
    sched.add(b)
    assert sched.decode_step == 0           # 正好在尝试点上
    seqs, is_prefill = sched.schedule()
    assert seqs, "排不出 prefill 时不能返回空步"
    assert not is_prefill
    assert a in seqs
    assert b not in seqs


def test_c2_all_dead_turns_into_empty_step():
    """(c2) waiting 里全是被 abort 的死请求：清完之后返回空步，不 assert。"""
    sched = make_scheduler()
    a = seq_of(8, max_tokens=4, tag=1)
    sched.add(a)
    sched.abort([a.seq_id])
    seqs, is_prefill = sched.schedule()
    assert seqs == [] and not is_prefill


def test_e_prefill_keeps_going():
    """(e) prefill 一旦开始就连着做：decode_step 清零后下一步仍在尝试点。"""
    sched = make_scheduler(max_num_batched_tokens=8)
    a, b, c = (seq_of(8, max_tokens=4, tag=i) for i in (1, 2, 3))
    for s in (a, b, c):
        sched.add(s)

    seqs, is_prefill = engine_step(sched)
    assert is_prefill and seqs == [a]
    assert sched.decode_step == 0

    seqs, is_prefill = sched.schedule()
    assert is_prefill, "prefill 要连着做，不能插一次 decode"
    assert seqs == [b]


def test_f_chunked_prefill_sits_last():
    """(f) 第一个请求就放不下：切一刀，它排在 batch 末尾且保持 WAITING。"""
    sched = make_scheduler(max_num_batched_tokens=4)
    a = seq_of(12, max_tokens=4, tag=1)
    sched.add(a)
    seqs, is_prefill = sched.schedule()
    assert is_prefill
    assert seqs == [a]
    assert a.num_scheduled_tokens == 4
    assert a.status == SequenceStatus.WAITING, "被切的请求还得留在 waiting 里等下一步"
    assert sched.waiting and sched.waiting[0] is a
