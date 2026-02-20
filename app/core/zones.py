# core/zones.py
from __future__ import annotations
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple

class ZoneIndex:
    def __init__(self, zones: List[dict]):
        # zones: [{"id": "...", "polygon": [(x,y), ...]}, ...]
        self.polys: Dict[str, Polygon] = {z["id"]: Polygon(z["polygon"]) for z in zones}

    @staticmethod
    def bbox_center(bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def in_zone(self, zone_id: str, bbox) -> bool:
        poly = self.polys.get(zone_id)
        if poly is None:
            return False
        cx, cy = self.bbox_center(bbox)
        return poly.contains(Point(cx, cy))
