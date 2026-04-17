# domain/models.py
"""
Modelos de dominio puros (sin dependencias externas pesadas).
Representan detecciones, tracks y estados de zona.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ZoneStatus(str, Enum):
    """Estado de un track respecto a una zona."""
    OUTSIDE = "outside"
    ENTERING = "entering"   # acumulando hits, aún no confirmado
    INSIDE = "inside"       # confirmado dentro
    EXITING = "exiting"     # dejó de verse, aún no confirmado salida


class TrackEventType(str, Enum):
    """Tipos de evento que genera el tracking por zona."""
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    ZONE_DWELL = "zone_dwell"       # permanencia prolongada
    INTRUSION = "intrusion"         # retrocompat con MVP


@dataclass(frozen=True)
class BBox:
    """Bounding box en píxeles (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, other: BBox) -> float:
        """Intersection over Union con otro BBox."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    """Una detección individual de un frame."""
    bbox: BBox
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    track_id: Optional[int] = None   # None si tracking deshabilitado


@dataclass
class ZoneTrackState:
    """Estado de un track específico respecto a una zona específica."""
    zone_id: str
    status: ZoneStatus = ZoneStatus.OUTSIDE
    hit_count: int = 0              # frames consecutivos dentro
    miss_count: int = 0             # frames consecutivos fuera (después de estar dentro)
    entered_at: Optional[float] = None   # timestamp de entrada confirmada
    last_seen_at: Optional[float] = None
    total_dwell_seconds: float = 0.0
    event_fired_enter: bool = False
    event_fired_exit: bool = False


@dataclass
class TrackState:
    """Estado completo de un track (una persona/objeto rastreado)."""
    track_id: int
    first_seen: float               # timestamp primera detección
    last_seen: float                 # timestamp última detección
    last_bbox: Optional[BBox] = None
    last_confidence: float = 0.0
    class_id: int = 0
    class_name: str = "person"
    frames_seen: int = 0
    is_active: bool = True
    zones: dict = field(default_factory=dict)  # zone_id -> ZoneTrackState

    def get_zone_state(self, zone_id: str) -> ZoneTrackState:
        if zone_id not in self.zones:
            self.zones[zone_id] = ZoneTrackState(zone_id=zone_id)
        return self.zones[zone_id]
