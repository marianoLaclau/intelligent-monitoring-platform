from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional

EventType = Literal["intrusion"]

class EventPayload(BaseModel):
    site_id: str
    camera_id: str
    ts: str  # ISO8601
    event_type: EventType
    zone_id: str
    confidence: float
    snapshot_path: Optional[str] = None
    meta: Dict[str, Any] = {}
