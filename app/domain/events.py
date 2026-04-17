# domain/events.py
"""
Esquemas de eventos del sistema.
Soporta eventos legacy (intrusion) y nuevos eventos de tracking (zone_enter, zone_exit, zone_dwell).
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional, List

# Tipos de evento soportados
EventType = Literal[
    "intrusion",        # retrocompat MVP
    "zone_enter",       # track confirmado dentro de zona
    "zone_exit",        # track confirmado fuera de zona
    "zone_dwell",       # track lleva mucho tiempo en zona
    "track_lost",       # track desapareció (timeout)
]


class EventPayload(BaseModel):
    """Payload unificado para todos los tipos de evento."""
    site_id: str
    camera_id: str
    ts: str                                     # ISO8601 UTC
    event_type: EventType
    zone_id: str
    confidence: float = 0.0
    track_id: Optional[int] = None              # ID del track (None en legacy)
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None             # para Fase 2
    meta: Dict[str, Any] = Field(default_factory=dict)


class TrackEvent(BaseModel):
    """Evento generado por el TrackStateManager (interno, antes de ser payload)."""
    event_type: EventType
    zone_id: str
    track_id: int
    confidence: float = 0.0
    dwell_seconds: float = 0.0
    meta: Dict[str, Any] = Field(default_factory=dict)
