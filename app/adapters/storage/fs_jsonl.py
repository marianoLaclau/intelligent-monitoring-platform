# adapters/storage/fs_jsonl.py
from __future__ import annotations
import os
import json

class JsonlEventStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, payload: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
