from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import AllocStatus, BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.prefill_attempt_interval = config.prefill_attempt_interval
        self.prefill_watermark = config.prefill_watermark
        self.stop_preempted_request = getattr(config, "stop_preempted_request", False)
        self.decode_step = 0
        # 每一步被抢占要重算的 token 数之和（2.3 的验收指标）
        self.recomputed_tokens = 0
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.finished: list[Sequence] = []
        self.preempted: list[int] = []

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        # 长度检查提前到入队：给 decode 至少留一个位置。
        if seq.num_prompt_tokens > self.max_model_len - 1:
            seq.status = SequenceStatus.FINISHED_IGNORED
        self.waiting.append(seq)

    def abort(self, seq_ids: list[int]):
        """只打标记，不动队列、不释放 block：这个请求可能还在 GPU 上跑。"""
        ids = set(seq_ids)
        for seq in list(self.waiting) + list(self.running):
            if seq.seq_id in ids and not seq.is_finished:
                seq.status = SequenceStatus.FINISHED_ABORTED

    def free_finished(self):
        """只出队，不释放 block。释放由 LLMEngine.step() 在 run 返回之后做。"""
        for seq in list(self.waiting):
            if seq.is_finished:
                self.waiting.remove(seq)
                self.finished.append(seq)
        for seq in list(self.running):
            if seq.is_finished:
                self.running.remove(seq)
                self.finished.append(seq)

    @property
    def kv_usage(self) -> float:
        """按 block 用量算，不按 token 数：水位线要卡的是显存，不是 token。"""
        total = len(self.block_manager.blocks)
        if not total:
            return 0.0
        return 1.0 - len(self.block_manager.free_block_ids) / total

    def _want_prefill(self) -> bool:
        """这一步要不要尝试 prefill。

        对回 2.2 的问题：原来只要 waiting 非空就先 prefill，prefill 了就不
        decode，于是任何一个新请求都会让所有 decode 停一步，停的时长等于
        prefill 的时长（generation stall）。这里用两个闸门限制它出现的频率
        和时机：尝试间隔 + KV 水位线。running 为空时不设闸门，否则会空转。
        """
        if not self.waiting:
            return False
        if not self.running:
            return True
        if self.decode_step % self.prefill_attempt_interval != 0:
            return False
        return self.kv_usage <= self.prefill_watermark

    def schedule(self) -> tuple[list[Sequence], bool]:
        self.free_finished()
        if self._want_prefill():
            scheduled_seqs = self._schedule_prefill()
            if scheduled_seqs:
                # prefill 一旦开始就连着做：清零后下一步仍在尝试点上，
                # 直到 waiting 清空 / running 满 / KV 超过水位线。
                self.decode_step = 0
                return scheduled_seqs, True
            # 排不出（放不下、或者队头全是死请求）：落到 decode，别返回空步
        scheduled_seqs = self._schedule_decode()
        if scheduled_seqs:
            self.decode_step += 1
            return scheduled_seqs, False
        return [], False

    def _schedule_prefill(self) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            num_cached_blocks = 0
            if not seq.block_table:
                status, num_cached_blocks = self.block_manager.can_allocate(seq)
                if status == AllocStatus.NEVER:
                    # 永远放不下：当场拒绝，别把队头堵住
                    self.waiting.popleft()
                    seq.status = SequenceStatus.FINISHED_IGNORED
                    self.finished.append(seq)
                    continue
                if status == AllocStatus.LATER:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 放不下就切一刀：切完立刻停，被切的这个排在 batch 末尾。
            # 每步最多切一个，且它在末尾 —— 2.7 的 ZOS 簿记靠这个前提。
            chunked = num_tokens > remaining
            if chunked:
                num_tokens = remaining
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = num_tokens
            num_batched_tokens += num_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            if chunked:
                break
        return scheduled_seqs

    def _schedule_decode(self) -> list[Sequence]:
        """按输出长度降序排：尾部分配 block，头部踢人（踢输出最长的）。

        为什么踢最长的：KV 满了以后空闲 block 只有两个来源 —— 有请求跑完，
        或者有请求被踢。踢一个快跑完的，等于把"马上要永久还回来的 block"
        换成"现在借出、以后还要拿回来，外加重算一遍"。输出最长的离跑完最近。
        """
        if not self.running:
            return []
        running = sorted(self.running, key=lambda s: s.num_completion_tokens, reverse=True)
        l, r = 0, len(running)
        scheduled = []
        while l < r and len(scheduled) < self.max_num_seqs:
            seq = running[r - 1]          # 尾部：输出最短的
            while not self.block_manager.can_append(seq):
                if r - l > 1:
                    self.preempt(running[l])   # 头部：输出最长的
                    l += 1
                else:
                    self.preempt(seq)          # 只剩自己还放不下，踢自己
                    r -= 1
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled.append(seq)
                r -= 1
        # 受 max_num_seqs 限制没轮到的要留在 running 里，不能丢
        self.running = deque(running[l:r] + scheduled)
        return scheduled

    def preempt(self, seq: Sequence):
        """两条出路，都要当场把 block 还回去。

        不立刻释放的话，同一步里紧跟着的 can_append 看不见这批 block，
        当前这个请求还是放不下，只好把自己也踢掉 —— 白踢一个人。
        """
        self.preempted.append(seq.seq_id)
        self.recomputed_tokens += seq.num_tokens
        seq.num_preemptions += 1
        self.block_manager.deallocate(seq)
        if self.stop_preempted_request:
            seq.status = SequenceStatus.FINISHED_PREEMPTED
            self.finished.append(seq)
            return
        seq.reset_for_recompute()
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """只判定结局，不出队、不释放。"""
        for seq, token_id in zip(seqs, token_ids):
            if seq.is_finished:  # abort 可能在本步 run 之前就打上了标记
                continue
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if seq.num_tokens >= self.max_model_len:
                seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            elif not seq.ignore_eos and token_id == self.eos:
                seq.status = SequenceStatus.FINISHED_STOPPED
            elif seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
