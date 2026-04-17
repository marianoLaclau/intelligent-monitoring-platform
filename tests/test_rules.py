# tests/test_rules.py
"""Tests para el motor de reglas."""
import pytest
from app.core.rules import IntrusionRule, RuleEngine
from app.domain.events import TrackEvent
from app.domain.models import TrackEventType


class TestIntrusionRuleLegacy:
    """Tests para la regla de intrusión legacy (sin tracking)."""

    def test_no_fire_below_min_hits(self):
        rule = IntrusionRule(zone_id="z1", min_hits=3, cooldown_seconds=0)
        assert rule.update(True) is False
        assert rule.update(True) is False

    def test_fire_at_min_hits(self):
        rule = IntrusionRule(zone_id="z1", min_hits=3, cooldown_seconds=0)
        rule.update(True)
        rule.update(True)
        assert rule.update(True) is True

    def test_reset_on_miss(self):
        rule = IntrusionRule(zone_id="z1", min_hits=3, cooldown_seconds=0)
        rule.update(True)
        rule.update(True)
        rule.update(False)  # reset
        assert rule.update(True) is False  # reinicia conteo

    def test_cooldown_blocks_refire(self):
        rule = IntrusionRule(zone_id="z1", min_hits=1, cooldown_seconds=9999)
        assert rule.update(True) is True
        # Dentro del cooldown, no debe volver a disparar
        assert rule.update(True) is False
        assert rule.update(True) is False


class TestRuleEngine:
    """Tests para el RuleEngine (con tracking)."""

    def _event(self, event_type, zone_id="z1", track_id=1):
        return TrackEvent(
            event_type=event_type,
            zone_id=zone_id,
            track_id=track_id,
            confidence=0.9,
        )

    def test_pass_through_enabled_events(self):
        engine = RuleEngine()
        events = [
            self._event(TrackEventType.ZONE_ENTER),
            self._event(TrackEventType.ZONE_EXIT),
        ]
        result = engine.process_events(events)
        assert len(result) == 2

    def test_filter_disabled_events(self):
        engine = RuleEngine()
        engine.disable_event_type(TrackEventType.ZONE_EXIT)

        events = [
            self._event(TrackEventType.ZONE_ENTER),
            self._event(TrackEventType.ZONE_EXIT),
        ]
        result = engine.process_events(events)
        assert len(result) == 1
        assert result[0].event_type == TrackEventType.ZONE_ENTER

    def test_suppressed_count(self):
        engine = RuleEngine()
        engine.disable_event_type(TrackEventType.ZONE_DWELL)

        events = [self._event(TrackEventType.ZONE_DWELL)] * 5
        engine.process_events(events)
        assert engine.suppressed_count == 5

    def test_enable_event_type(self):
        engine = RuleEngine()
        engine.disable_event_type(TrackEventType.ZONE_ENTER)
        engine.enable_event_type(TrackEventType.ZONE_ENTER)

        events = [self._event(TrackEventType.ZONE_ENTER)]
        result = engine.process_events(events)
        assert len(result) == 1

    def test_zone_rules_enrichment(self):
        engine = RuleEngine(zone_rules={"z1": {"zone_type": "restricted"}})
        events = [self._event(TrackEventType.ZONE_ENTER, zone_id="z1")]
        result = engine.process_events(events)
        assert result[0].meta.get("zone_rule") == "restricted"

    def test_empty_events(self):
        engine = RuleEngine()
        result = engine.process_events([])
        assert result == []
