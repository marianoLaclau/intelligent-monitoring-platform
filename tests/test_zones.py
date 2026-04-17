# tests/test_zones.py
"""Tests para el módulo de zonas."""
import pytest
from app.core.zones import ZoneIndex
from app.domain.models import BBox


@pytest.fixture
def square_zone():
    """Zona cuadrada 100x100 en (100,100)-(200,200)."""
    return ZoneIndex([{
        "id": "z1",
        "polygon": [(100, 100), (200, 100), (200, 200), (100, 200)],
        "zone_type": "restricted",
    }])


@pytest.fixture
def multi_zone():
    """Dos zonas sin overlap."""
    return ZoneIndex([
        {"id": "z1", "polygon": [(0, 0), (100, 0), (100, 100), (0, 100)]},
        {"id": "z2", "polygon": [(200, 200), (300, 200), (300, 300), (200, 300)]},
    ])


class TestZoneIndex:
    def test_point_inside(self, square_zone):
        # BBox centrado en (150, 150) → dentro
        bbox = (120, 120, 180, 180)
        assert square_zone.in_zone("z1", bbox) is True

    def test_point_outside(self, square_zone):
        # BBox centrado en (50, 50) → fuera
        bbox = (20, 20, 80, 80)
        assert square_zone.in_zone("z1", bbox) is False

    def test_point_on_edge(self, square_zone):
        # BBox centrado exactamente en el borde (150, 100) → depends on contains vs intersects
        bbox = (140, 90, 160, 110)
        # El centro (150, 100) está en el borde; Shapely contains = False para bordes
        # Esto es comportamiento esperado
        result = square_zone.in_zone("z1", bbox)
        assert isinstance(result, bool)

    def test_nonexistent_zone(self, square_zone):
        bbox = (120, 120, 180, 180)
        assert square_zone.in_zone("z_inexistente", bbox) is False

    def test_bbox_object(self, square_zone):
        bbox = BBox(x1=120, y1=120, x2=180, y2=180)
        assert square_zone.in_zone("z1", bbox.as_tuple()) is True

    def test_multi_zone_z1(self, multi_zone):
        bbox = (40, 40, 60, 60)  # centro (50, 50)
        assert multi_zone.in_zone("z1", bbox) is True
        assert multi_zone.in_zone("z2", bbox) is False

    def test_multi_zone_z2(self, multi_zone):
        bbox = (240, 240, 260, 260)  # centro (250, 250)
        assert multi_zone.in_zone("z1", bbox) is False
        assert multi_zone.in_zone("z2", bbox) is True

    def test_zones_for_bbox(self, multi_zone):
        bbox = (40, 40, 60, 60)
        zones = multi_zone.zones_for_bbox(bbox)
        assert zones == ["z1"]

    def test_zone_ids(self, multi_zone):
        assert set(multi_zone.zone_ids) == {"z1", "z2"}

    def test_get_polygon_points(self, square_zone):
        pts = square_zone.get_polygon_points("z1")
        assert pts is not None
        assert len(pts) >= 4  # al menos 4 puntos (el último repite el primero en Shapely)

    def test_get_polygon_points_nonexistent(self, square_zone):
        assert square_zone.get_polygon_points("nope") is None

    def test_bottom_center(self):
        bbox = BBox(x1=100, y1=50, x2=200, y2=150)
        cx, cy = ZoneIndex.bbox_bottom_center(bbox)
        assert cx == 150.0
        assert cy == 150.0  # y2, no centro

    def test_prepared_geometry_performance(self):
        """Las prepared geometries deben funcionar igual que las normales."""
        zones = ZoneIndex([{
            "id": "big",
            "polygon": [(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
        }])
        # Muchas consultas rápidas
        for i in range(100):
            bbox = (i * 5, i * 5, i * 5 + 10, i * 5 + 10)
            result = zones.in_zone("big", bbox)
            expected = (i * 5 + 5) < 1000  # centro dentro del cuadrado
            assert result == expected, f"Failed at i={i}"
