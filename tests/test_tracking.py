# tests/test_tracking.py
"""Tests para el TrackStateManager."""
import pytest
import time
from app.core.tracking import TrackStateManager
from app.core.zones import ZoneIndex
from app.domain.models import Detection, BBox, TrackEventType


@pytest.fixture
def zone_index():
    """Zona cuadrada simple."""
    return ZoneIndex([{
        "id": "z1",
        "polygon": [(100, 100), (300, 100), (300, 300), (100, 300)],
    }])


@pytest.fixture
def manager():
    """TrackStateManager con parámetros de test."""
    return TrackStateManager(
        min_hits_enter=3,
        max_misses_exit=5,
        track_timeout_seconds=2.0,
        dwell_alert_seconds=10.0,
        cooldown_per_track_seconds=0.0,  # sin cooldown para tests
    )


def _det(track_id: int, cx: float, cy: float, conf: float = 0.8) -> Detection:
    """Helper: crea Detection con centro en (cx, cy)."""
    return Detection(
        bbox=BBox(x1=cx - 20, y1=cy - 40, x2=cx + 20, y2=cy + 40),
        confidence=conf,
        class_id=0,
        class_name="person",
        track_id=track_id,
    )


class TestTrackStateManager:
    def test_no_detections_no_events(self, manager, zone_index):
        events = manager.update([], zone_index, ["z1"], now=1.0)
        assert events == []

    def test_detection_outside_zone_no_events(self, manager, zone_index):
        det = _det(track_id=1, cx=50, cy=50)  # fuera de z1
        events = manager.update([det], zone_index, ["z1"], now=1.0)
        assert events == []

    def test_zone_enter_after_min_hits(self, manager, zone_index):
        """Después de min_hits=3 frames dentro, debe emitir zone_enter."""
        det = _det(track_id=1, cx=200, cy=200)  # dentro de z1

        all_events = []
        for i in range(5):
            events = manager.update([det], zone_index, ["z1"], now=1.0 + i * 0.1)
            all_events.extend(events)

        event_types = [e.event_type for e in all_events]
        assert TrackEventType.ZONE_ENTER in event_types
        assert TrackEventType.INTRUSION in event_types  # retrocompat

    def test_no_enter_before_min_hits(self, manager, zone_index):
        """Antes de min_hits, no debe emitir nada."""
        det = _det(track_id=1, cx=200, cy=200)

        events = manager.update([det], zone_index, ["z1"], now=1.0)
        events += manager.update([det], zone_index, ["z1"], now=1.1)
        # Solo 2 frames, min_hits=3 → sin evento
        assert len(events) == 0

    def test_zone_exit_after_max_misses(self, manager, zone_index):
        """Después de entrar, si desaparece de zona, emite zone_exit."""
        det_in = _det(track_id=1, cx=200, cy=200)
        det_out = _det(track_id=1, cx=50, cy=50)

        t = 1.0
        # Entrar (min_hits=3)
        for i in range(4):
            manager.update([det_in], zone_index, ["z1"], now=t)
            t += 0.1

        # Salir (max_misses=5)
        all_events = []
        for i in range(10):
            events = manager.update([det_out], zone_index, ["z1"], now=t)
            all_events.extend(events)
            t += 0.1

        event_types = [e.event_type for e in all_events]
        assert TrackEventType.ZONE_EXIT in event_types

    def test_zone_exit_includes_dwell_time(self, manager, zone_index):
        """El evento zone_exit debe incluir dwell_seconds."""
        det_in = _det(track_id=1, cx=200, cy=200)
        det_out = _det(track_id=1, cx=50, cy=50)

        t = 100.0
        for i in range(4):
            manager.update([det_in], zone_index, ["z1"], now=t)
            t += 1.0

        # Dwell = ~4 seconds so far
        all_events = []
        for i in range(10):
            events = manager.update([det_out], zone_index, ["z1"], now=t)
            all_events.extend(events)
            t += 1.0

        exits = [e for e in all_events if e.event_type == TrackEventType.ZONE_EXIT]
        assert len(exits) >= 1
        assert exits[0].dwell_seconds > 0

    def test_multiple_tracks_independent(self, manager, zone_index):
        """Dos tracks independientes deben generar eventos separados."""
        det1 = _det(track_id=1, cx=200, cy=200)  # dentro
        det2 = _det(track_id=2, cx=50, cy=50)     # fuera

        all_events = []
        for i in range(5):
            events = manager.update([det1, det2], zone_index, ["z1"], now=1.0 + i * 0.1)
            all_events.extend(events)

        # Solo track 1 debe tener zone_enter
        enters = [e for e in all_events if e.event_type == TrackEventType.ZONE_ENTER]
        assert all(e.track_id == 1 for e in enters)

    def test_track_lost_emits_exit(self, manager, zone_index):
        """Si un track desaparece (timeout), debe emitir exit de zonas."""
        det = _det(track_id=1, cx=200, cy=200)

        t = 1.0
        for i in range(4):
            manager.update([det], zone_index, ["z1"], now=t)
            t += 0.1

        # Track desaparece por track_timeout_seconds=2.0
        all_events = []
        for i in range(30):
            events = manager.update([], zone_index, ["z1"], now=t)
            all_events.extend(events)
            t += 0.2

        event_types = [e.event_type for e in all_events]
        assert TrackEventType.ZONE_EXIT in event_types

    def test_unique_count(self, manager, zone_index):
        """Conteo de tracks únicos por zona."""
        t = 1.0
        for tid in [1, 2, 3]:
            det = _det(track_id=tid, cx=200, cy=200)
            for i in range(2):
                manager.update([det], zone_index, ["z1"], now=t)
                t += 0.1

        assert manager.unique_count_for_zone("z1") == 3

    def test_tracks_in_zone(self, manager, zone_index):
        """tracks_in_zone debe retornar solo los que están INSIDE."""
        det = _det(track_id=1, cx=200, cy=200)

        # Antes de min_hits, no está "inside"
        manager.update([det], zone_index, ["z1"], now=1.0)
        assert manager.tracks_in_zone("z1") == []

        # Después de min_hits
        for i in range(4):
            manager.update([det], zone_index, ["z1"], now=1.1 + i * 0.1)
        assert 1 in manager.tracks_in_zone("z1")

    def test_cooldown_suppresses_repeated_enter(self, zone_index):
        """Cooldown debe evitar que el mismo track genere enters repetidos."""
        mgr = TrackStateManager(
            min_hits_enter=2,
            max_misses_exit=3,
            track_timeout_seconds=10.0,
            dwell_alert_seconds=999.0,
            cooldown_per_track_seconds=5.0,  # 5 sec cooldown
        )

        det_in = _det(track_id=1, cx=200, cy=200)
        det_out = _det(track_id=1, cx=50, cy=50)

        t = 1.0
        # Primera entrada
        enters = []
        for i in range(5):
            events = mgr.update([det_in], zone_index, ["z1"], now=t)
            enters.extend([e for e in events if e.event_type == TrackEventType.ZONE_ENTER])
            t += 0.1
        assert len(enters) == 1

        # Salir
        for i in range(5):
            mgr.update([det_out], zone_index, ["z1"], now=t)
            t += 0.1

        # Intentar entrar de nuevo rápido (dentro del cooldown de 5s)
        enters2 = []
        for i in range(5):
            events = mgr.update([det_in], zone_index, ["z1"], now=t)
            enters2.extend([e for e in events if e.event_type == TrackEventType.ZONE_ENTER])
            t += 0.1
        # Debería estar suprimido por cooldown (t total < 5s)
        assert len(enters2) == 0

    def test_cleanup_old_tracks(self, manager, zone_index):
        det = _det(track_id=1, cx=200, cy=200)
        manager.update([det], zone_index, ["z1"], now=1.0)

        # Marcar como inactivo
        manager._tracks[1].is_active = False
        manager._tracks[1].last_seen = 1.0

        # Limpiar con max_age=5s, now=100
        cleaned = manager.cleanup_old_tracks(max_age_seconds=5.0, now=100.0)
        assert cleaned == 1
        assert 1 not in manager._tracks

    def test_reset(self, manager, zone_index):
        det = _det(track_id=1, cx=200, cy=200)
        manager.update([det], zone_index, ["z1"], now=1.0)
        assert manager.track_count > 0

        manager.reset()
        assert manager.track_count == 0

    def test_detection_without_track_id_ignored(self, manager, zone_index):
        """Detecciones sin track_id deben ser ignoradas."""
        det = Detection(
            bbox=BBox(x1=180, y1=160, x2=220, y2=240),
            confidence=0.9,
            track_id=None,
        )
        events = manager.update([det], zone_index, ["z1"], now=1.0)
        assert events == []
        assert manager.track_count == 0

    def test_dwell_alert(self, zone_index):
        """Debe emitir zone_dwell después de dwell_alert_seconds."""
        mgr = TrackStateManager(
            min_hits_enter=2,
            max_misses_exit=5,
            track_timeout_seconds=999.0,
            dwell_alert_seconds=5.0,  # alert after 5 seconds
            cooldown_per_track_seconds=0.0,
        )

        det = _det(track_id=1, cx=200, cy=200)
        all_events = []
        t = 1.0
        for i in range(100):  # simulate ~10 seconds
            events = mgr.update([det], zone_index, ["z1"], now=t)
            all_events.extend(events)
            t += 0.1

        dwell_events = [e for e in all_events if e.event_type == TrackEventType.ZONE_DWELL]
        assert len(dwell_events) >= 1
        assert dwell_events[0].dwell_seconds >= 5.0


class TestBBox:
    def test_center(self):
        bbox = BBox(x1=100, y1=100, x2=200, y2=200)
        assert bbox.center == (150.0, 150.0)

    def test_area(self):
        bbox = BBox(x1=0, y1=0, x2=10, y2=10)
        assert bbox.area == 100.0

    def test_iou_identical(self):
        bbox = BBox(x1=0, y1=0, x2=10, y2=10)
        assert bbox.iou(bbox) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        b1 = BBox(x1=0, y1=0, x2=10, y2=10)
        b2 = BBox(x1=20, y1=20, x2=30, y2=30)
        assert b1.iou(b2) == pytest.approx(0.0)

    def test_iou_partial(self):
        b1 = BBox(x1=0, y1=0, x2=10, y2=10)
        b2 = BBox(x1=5, y1=5, x2=15, y2=15)
        # inter = 5x5 = 25, union = 100 + 100 - 25 = 175
        assert b1.iou(b2) == pytest.approx(25 / 175)

    def test_as_tuple(self):
        bbox = BBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0)
        assert bbox.as_tuple() == (1.0, 2.0, 3.0, 4.0)
