# core/rules.py
"""
Motor de reglas.

Con tracking habilitado:
  - TrackStateManager emite eventos (zone_enter, zone_exit, zone_dwell)
  - RuleEngine filtra/enriquece según configuración de reglas
  
Sin tracking (legacy):
  - IntrusionRule funciona como antes (por frame, sin track_id)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from app.domain.events import TrackEvent
from app.domain.models import TrackEventType


class IntrusionRule:
    """
    Regla de intrusión legacy (sin tracking).
    Mantiene compatibilidad con el MVP original.
    """

    def __init__(self, zone_id: str, min_hits: int = 3, cooldown_seconds: int = 20):
        self.zone_id = zone_id
        self.min_hits = min_hits
        self.cooldown_seconds = cooldown_seconds

        self._hit_count = 0
        self._last_fire_ts: float = 0.0

    def update(self, is_in_zone: bool) -> bool:
        now = time.time()

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


class RuleEngine:
    """
    Motor de reglas que procesa eventos de tracking y decide cuáles emitir.
    
    Responsabilidades:
      - Filtrar eventos según tipo habilitado
      - Aplicar reglas específicas por zona (dwell_alert_seconds por zona)
      - Agrupar/suprimir eventos redundantes
      - Enriquecer metadata
    """

    def __init__(self, zone_rules: Optional[Dict[str, dict]] = None):
        """
        Args:
            zone_rules: {zone_id: {"cooldown_seconds": 20, "dwell_alert_seconds": 60, ...}}
        """
        self.zone_rules: Dict[str, dict] = zone_rules or {}
        self._suppressed_count: int = 0

        # Tipos de evento habilitados
        self.enabled_events = {
            TrackEventType.ZONE_ENTER,
            TrackEventType.ZONE_EXIT,
            TrackEventType.ZONE_DWELL,
            TrackEventType.INTRUSION,
        }

    def process_events(self, events: List[TrackEvent]) -> List[TrackEvent]:
        """
        Filtra y enriquece una lista de eventos crudos del TrackStateManager.
        
        Returns:
            Lista filtrada de eventos que deben generar evidencia + notificación.
        """
        output: List[TrackEvent] = []

        for ev in events:
            # Filtrar por tipo habilitado
            if ev.event_type not in self.enabled_events:
                self._suppressed_count += 1
                continue

            # Aplicar reglas específicas de zona
            zr = self.zone_rules.get(ev.zone_id, {})

            # Enriquecer metadata con info de regla
            ev.meta["zone_rule"] = zr.get("zone_type", "restricted")

            output.append(ev)

        return output

    def disable_event_type(self, event_type: TrackEventType) -> None:
        self.enabled_events.discard(event_type)

    def enable_event_type(self, event_type: TrackEventType) -> None:
        self.enabled_events.add(event_type)

    @property
    def suppressed_count(self) -> int:
        return self._suppressed_count
