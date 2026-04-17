# core/tracking.py
"""
TrackStateManager — Gestión de estado de tracks por zona.

Responsabilidades:
  - Mantener el estado de cada track (activo, perdido, eliminado)
  - Evaluar transiciones de zona por track (OUTSIDE → ENTERING → INSIDE → EXITING → OUTSIDE)
  - Emitir eventos: zone_enter, zone_exit, zone_dwell, track_lost
  - Manejar timeouts y limpieza de tracks antiguos
  - Conteo de personas únicas por zona

Diseño: el tracker de BoT-SORT/ByteTrack (en Ultralytics) asigna track_id.
Este módulo NO hace tracking visual — solo gestiona el estado lógico.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from loguru import logger

from app.domain.models import (
    Detection,
    TrackState,
    ZoneTrackState,
    ZoneStatus,
    TrackEventType,
)
from app.domain.events import TrackEvent
from app.core.zones import ZoneIndex


class TrackStateManager:
    """
    Administra el ciclo de vida de tracks y sus transiciones por zona.
    
    Flujo por track por zona:
      OUTSIDE ──[min_hits_enter frames dentro]──► INSIDE  → emite zone_enter
      INSIDE  ──[max_misses_exit frames fuera]──► OUTSIDE → emite zone_exit
      INSIDE  ──[dwell_alert_seconds]───────────► INSIDE  → emite zone_dwell (una vez)
    
    Flujo de track global:
      ACTIVO  ──[track_timeout_seconds sin ver]─► PERDIDO → emite track_lost + zone_exit pendientes
    """

    def __init__(
        self,
        min_hits_enter: int = 3,
        max_misses_exit: int = 15,
        track_timeout_seconds: float = 10.0,
        dwell_alert_seconds: float = 60.0,
        cooldown_per_track_seconds: float = 20.0,
    ):
        self.min_hits_enter = min_hits_enter
        self.max_misses_exit = max_misses_exit
        self.track_timeout_seconds = track_timeout_seconds
        self.dwell_alert_seconds = dwell_alert_seconds
        self.cooldown_per_track_seconds = cooldown_per_track_seconds

        # Estado principal
        self._tracks: Dict[int, TrackState] = {}

        # Cooldown por (track_id, zone_id) → timestamp último evento
        self._cooldowns: Dict[Tuple[int, str], float] = {}

        # Métricas
        self._unique_tracks_per_zone: Dict[str, Set[int]] = defaultdict(set)
        self._total_events: int = 0

    @property
    def active_tracks(self) -> Dict[int, TrackState]:
        return {tid: t for tid, t in self._tracks.items() if t.is_active}

    @property
    def track_count(self) -> int:
        return len(self.active_tracks)

    def unique_count_for_zone(self, zone_id: str) -> int:
        return len(self._unique_tracks_per_zone.get(zone_id, set()))

    def tracks_in_zone(self, zone_id: str) -> List[int]:
        """Track IDs actualmente INSIDE en una zona."""
        result = []
        for tid, ts in self._tracks.items():
            if not ts.is_active:
                continue
            zs = ts.zones.get(zone_id)
            if zs and zs.status == ZoneStatus.INSIDE:
                result.append(tid)
        return result

    def update(
        self,
        detections: List[Detection],
        zone_index: ZoneIndex,
        zone_ids: List[str],
        now: Optional[float] = None,
    ) -> List[TrackEvent]:
        """
        Actualiza el estado con las detecciones del frame actual.
        
        Args:
            detections: Lista de Detection (con track_id asignado por YOLO tracker)
            zone_index: Índice de zonas (polígonos)
            zone_ids: IDs de zonas a evaluar
            now: timestamp actual (default: time.time())
            
        Returns:
            Lista de TrackEvent generados en este frame
        """
        now = now or time.time()
        events: List[TrackEvent] = []

        # Track IDs vistos en este frame
        seen_track_ids: Set[int] = set()

        # ── 1. Procesar detecciones con track_id ──
        for det in detections:
            if det.track_id is None:
                continue  # sin tracking, ignorar

            tid = det.track_id
            seen_track_ids.add(tid)

            # Crear o actualizar TrackState
            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    track_id=tid,
                    first_seen=now,
                    last_seen=now,
                    last_bbox=det.bbox,
                    last_confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    frames_seen=1,
                    is_active=True,
                )
            else:
                ts = self._tracks[tid]
                ts.last_seen = now
                ts.last_bbox = det.bbox
                ts.last_confidence = det.confidence
                ts.frames_seen += 1
                ts.is_active = True

            track = self._tracks[tid]

            # ── 2. Evaluar cada zona para este track ──
            for zone_id in zone_ids:
                is_inside = zone_index.in_zone(zone_id, det.bbox.as_tuple())
                zone_events = self._update_zone_state(track, zone_id, is_inside, now)
                events.extend(zone_events)

                # Registrar track como visto en zona (para conteo único)
                if is_inside:
                    self._unique_tracks_per_zone[zone_id].add(tid)

        # ── 3. Manejar tracks NO vistos en este frame ──
        for tid, track in list(self._tracks.items()):
            if tid in seen_track_ids:
                continue
            if not track.is_active:
                continue

            elapsed = now - track.last_seen

            # Incrementar miss_count en zonas donde estaba INSIDE
            for zone_id in zone_ids:
                zs = track.get_zone_state(zone_id)
                if zs.status in (ZoneStatus.INSIDE, ZoneStatus.ENTERING):
                    zs.miss_count += 1
                    zs.hit_count = 0

                    # Check EXIT
                    if zs.miss_count >= self.max_misses_exit and zs.status == ZoneStatus.INSIDE:
                        exit_events = self._fire_zone_exit(track, zs, now)
                        events.extend(exit_events)
                    elif zs.status == ZoneStatus.ENTERING:
                        # No llegó a confirmar entrada, volver a OUTSIDE
                        if zs.miss_count > self.min_hits_enter:
                            zs.status = ZoneStatus.OUTSIDE
                            zs.hit_count = 0
                            zs.miss_count = 0

            # Check timeout global del track
            if elapsed >= self.track_timeout_seconds:
                lost_events = self._handle_track_lost(track, zone_ids, now)
                events.extend(lost_events)

        self._total_events += len(events)
        return events

    def _update_zone_state(
        self,
        track: TrackState,
        zone_id: str,
        is_inside: bool,
        now: float,
    ) -> List[TrackEvent]:
        """Máquina de estados para un track en una zona."""
        events: List[TrackEvent] = []
        zs = track.get_zone_state(zone_id)
        zs.last_seen_at = now

        if is_inside:
            zs.hit_count += 1
            zs.miss_count = 0

            if zs.status == ZoneStatus.OUTSIDE or zs.status == ZoneStatus.ENTERING:
                # Acumulando hits para confirmar entrada
                zs.status = ZoneStatus.ENTERING

                if zs.hit_count >= self.min_hits_enter:
                    # ¡Entrada confirmada!
                    zs.status = ZoneStatus.INSIDE
                    zs.entered_at = now
                    zs.event_fired_enter = False
                    zs.event_fired_exit = False

                    # Emitir zone_enter (con cooldown)
                    if self._check_cooldown(track.track_id, zone_id, now):
                        events.append(TrackEvent(
                            event_type=TrackEventType.ZONE_ENTER,
                            zone_id=zone_id,
                            track_id=track.track_id,
                            confidence=track.last_confidence,
                            meta={
                                "class_name": track.class_name,
                                "frames_seen": track.frames_seen,
                            },
                        ))
                        # También emitir "intrusion" para retrocompat
                        events.append(TrackEvent(
                            event_type=TrackEventType.INTRUSION,
                            zone_id=zone_id,
                            track_id=track.track_id,
                            confidence=track.last_confidence,
                            meta={
                                "class_name": track.class_name,
                                "frames_seen": track.frames_seen,
                                "source": "tracking",
                            },
                        ))
                        zs.event_fired_enter = True
                        self._set_cooldown(track.track_id, zone_id, now)

            elif zs.status == ZoneStatus.INSIDE:
                # Ya está dentro — check dwell time
                if zs.entered_at is not None:
                    zs.total_dwell_seconds = now - zs.entered_at

                    if (
                        zs.total_dwell_seconds >= self.dwell_alert_seconds
                        and not zs.event_fired_exit  # usar como flag de "dwell ya emitido"
                    ):
                        events.append(TrackEvent(
                            event_type=TrackEventType.ZONE_DWELL,
                            zone_id=zone_id,
                            track_id=track.track_id,
                            confidence=track.last_confidence,
                            dwell_seconds=zs.total_dwell_seconds,
                            meta={
                                "class_name": track.class_name,
                                "alert": "prolonged_presence",
                            },
                        ))
                        zs.event_fired_exit = True  # reusar flag para no repetir

            elif zs.status == ZoneStatus.EXITING:
                # Volvió a entrar antes de confirmar salida
                zs.status = ZoneStatus.INSIDE
                zs.miss_count = 0

        else:
            # Fuera de zona
            zs.miss_count += 1
            zs.hit_count = 0

            if zs.status == ZoneStatus.INSIDE:
                # Empezar a contar misses
                if zs.miss_count >= self.max_misses_exit:
                    exit_events = self._fire_zone_exit(track, zs, now)
                    events.extend(exit_events)

            elif zs.status == ZoneStatus.ENTERING:
                # No confirmó entrada, resetear
                if zs.miss_count > self.min_hits_enter:
                    zs.status = ZoneStatus.OUTSIDE
                    zs.hit_count = 0
                    zs.miss_count = 0

        return events

    def _fire_zone_exit(
        self,
        track: TrackState,
        zs: ZoneTrackState,
        now: float,
    ) -> List[TrackEvent]:
        """Genera evento de salida de zona."""
        events: List[TrackEvent] = []

        dwell = 0.0
        if zs.entered_at is not None:
            dwell = now - zs.entered_at

        zs.status = ZoneStatus.OUTSIDE
        zs.hit_count = 0
        zs.miss_count = 0
        zs.total_dwell_seconds = dwell

        if zs.event_fired_enter:  # solo emitir EXIT si emitimos ENTER
            events.append(TrackEvent(
                event_type=TrackEventType.ZONE_EXIT,
                zone_id=zs.zone_id,
                track_id=track.track_id,
                confidence=track.last_confidence,
                dwell_seconds=dwell,
                meta={
                    "class_name": track.class_name,
                },
            ))

        zs.entered_at = None
        zs.event_fired_enter = False
        zs.event_fired_exit = False

        return events

    def _handle_track_lost(
        self,
        track: TrackState,
        zone_ids: List[str],
        now: float,
    ) -> List[TrackEvent]:
        """Maneja un track que desapareció (timeout)."""
        events: List[TrackEvent] = []

        # Cerrar todas las zonas donde estaba INSIDE
        for zone_id in zone_ids:
            zs = track.get_zone_state(zone_id)
            if zs.status in (ZoneStatus.INSIDE, ZoneStatus.ENTERING):
                exit_events = self._fire_zone_exit(track, zs, now)
                events.extend(exit_events)

        track.is_active = False

        logger.debug(
            f"Track {track.track_id} lost after "
            f"{now - track.last_seen:.1f}s (seen {track.frames_seen} frames)"
        )
        return events

    def _check_cooldown(self, track_id: int, zone_id: str, now: float) -> bool:
        """Verifica si pasó el cooldown para este track+zona."""
        key = (track_id, zone_id)
        last = self._cooldowns.get(key, 0.0)
        return (now - last) >= self.cooldown_per_track_seconds

    def _set_cooldown(self, track_id: int, zone_id: str, now: float) -> None:
        self._cooldowns[(track_id, zone_id)] = now

    def get_stats(self) -> dict:
        """Estadísticas del tracking para monitoreo/debug."""
        active = self.active_tracks
        return {
            "total_tracks_seen": len(self._tracks),
            "active_tracks": len(active),
            "total_events_emitted": self._total_events,
            "unique_per_zone": {
                zid: len(tids) for zid, tids in self._unique_tracks_per_zone.items()
            },
            "tracks_in_zones": {
                zid: self.tracks_in_zone(zid)
                for zid in self._unique_tracks_per_zone.keys()
            },
        }

    def cleanup_old_tracks(self, max_age_seconds: float = 300.0, now: Optional[float] = None) -> int:
        """Limpia tracks inactivos muy antiguos para liberar memoria."""
        now = now or time.time()
        to_remove = [
            tid for tid, ts in self._tracks.items()
            if not ts.is_active and (now - ts.last_seen) > max_age_seconds
        ]
        for tid in to_remove:
            del self._tracks[tid]
        return len(to_remove)

    def reset(self) -> None:
        """Resetea todo el estado (útil al cambiar de fuente)."""
        self._tracks.clear()
        self._cooldowns.clear()
        self._unique_tracks_per_zone.clear()
        self._total_events = 0
        logger.info("TrackStateManager reset")
