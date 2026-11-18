"""每步一行 JSONL。挂到 engine.trace 上，引擎每步调一次。

只写 StepRecord 里现有的字段，不 import torch。
"""
import json
from dataclasses import asdict
from pathlib import Path


class TraceWriter:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")

    def __call__(self, record):
        row = asdict(record)
        row["gpu_busy"] = record.gpu_busy
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self):
        if not self.f.closed:
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
