# core/rules.py
from __future__ import annotations
import time
from typing import Optional

class IntrusionRule:
    def __init__(self, zone_id: str, min_hits: int, cooldown_seconds: int):
        self.zone_id = zone_id
        self.min_hits = min_hits
        self.cooldown_seconds = cooldown_seconds

        self._hit_count = 0
        self._last_fire_ts: float = 0.0

    def update(self, is_in_zone: bool) -> bool:
        now = time.time()

        # cooldown
        if (now - self._last_fire_ts) < self.cooldown_seconds:
            self._hit_count = 0 if not is_in_zone else self._hit_count
            return False

        if is_in_zone:
            self._hit_count += 1
        else:
            self._hit_count = 0

        if self._hit_count >= self.min_hits:
            self._last_fire_ts = now
            self._hit_count = 0
            return True

        return False
