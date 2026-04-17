# core/zones.py
"""
Gestión de zonas con Shapely.
Evalúa si un punto/bbox está dentro de un polígono definido.
"""
from __future__ import annotations

from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from typing import Dict, List, Optional, Set, Tuple

from app.domain.models import BBox


class ZoneIndex:
    """
    Índice espacial de zonas.
    Usa prepared geometries de Shapely para consultas rápidas.
    """

    def __init__(self, zones: List[dict]):
        """
        Args:
            zones: [{"id": "zona_01", "polygon": [(x,y), ...], "zone_type": "restricted"}, ...]
        """
        self.polys: Dict[str, Polygon] = {}
        self._prepared: Dict[str, object] = {}
        self.zone_types: Dict[str, str] = {}

        for z in zones:
            zid = z["id"]
            poly = Polygon(z["polygon"])
            self.polys[zid] = poly
            self._prepared[zid] = prep(poly)  # prepared geometry = consultas ~5x más rápidas
            self.zone_types[zid] = z.get("zone_type", "restricted")

    @property
    def zone_ids(self) -> List[str]:
        return list(self.polys.keys())

    @staticmethod
    def bbox_center(bbox) -> Tuple[float, float]:
        """Centro de un bbox (acepta tuple o BBox)."""
        if isinstance(bbox, BBox):
            return bbox.center
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def bbox_bottom_center(bbox) -> Tuple[float, float]:
        """Punto inferior central (pies de la persona). Mejor para intrusión."""
        if isinstance(bbox, BBox):
            return ((bbox.x1 + bbox.x2) / 2.0, bbox.y2)
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    def in_zone(self, zone_id: str, bbox, use_bottom: bool = False) -> bool:
        """
        Evalúa si el punto de referencia del bbox está dentro de la zona.
        
        Args:
            zone_id: ID de la zona
            bbox: BBox object o tuple (x1, y1, x2, y2)
            use_bottom: Si True, usa el punto inferior central (pies).
                        Si False, usa el centro del bbox.
        """
        prepared = self._prepared.get(zone_id)
        if prepared is None:
            return False

        if use_bottom:
            cx, cy = self.bbox_bottom_center(bbox)
        else:
            cx, cy = self.bbox_center(bbox)

        return prepared.contains(Point(cx, cy))

    def zones_for_bbox(self, bbox, use_bottom: bool = False) -> List[str]:
        """Retorna todas las zonas que contienen el bbox."""
        return [zid for zid in self.polys if self.in_zone(zid, bbox, use_bottom)]

    def get_polygon_points(self, zone_id: str) -> Optional[List[Tuple[int, int]]]:
        """Retorna los puntos del polígono como lista de (x, y) enteros (para dibujar)."""
        poly = self.polys.get(zone_id)
        if poly is None:
            return None
        return [(int(x), int(y)) for x, y in poly.exterior.coords]
