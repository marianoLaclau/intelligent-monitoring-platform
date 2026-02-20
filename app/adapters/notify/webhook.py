# adapters/notify/webhook.py
from __future__ import annotations
from typing import Optional
import requests

class WebhookNotifier:
    def __init__(self, url: str, timeout_seconds: int = 5):
        self.url = url
        self.timeout = timeout_seconds

    def send(self, payload: dict) -> Optional[int]:
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        return r.status_code
