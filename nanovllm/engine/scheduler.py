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
        self.decode_step = 0
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
        scheduled_seqs = []
        if not self.running:
            return scheduled_seqs
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_preemptions += 1
        self.preempted.append(seq.seq_id)
        self.block_manager.deallocate(seq)
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
