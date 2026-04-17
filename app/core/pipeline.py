# core/pipeline.py
"""
Pipeline v6 — Threading correcto para tiempo real.

ARQUITECTURA:
  ┌─────────────────────────────────────────────────────┐
  │  MAIN THREAD (30 fps)                               │
  │  for frame in source:                               │
  │    1. mostrar frame + últimas detecciones (fluido)  │
  │    2. cada N frames → copiar al slot compartido     │
  │    3. procesar eventos pendientes                   │
  │    4. sleep para mantener fps del source             │
  └───────────────────┬─────────────────────────────────┘
                      │ frame cada N (regular)
  ┌───────────────────▼─────────────────────────────────┐
  │  INFERENCE THREAD                                    │
  │  while running:                                      │
  │    1. esperar frame nuevo                           │
  │    2. model.track(frame, persist=True)              │
  │    3. publicar detecciones                          │
  │    4. evaluar zonas → generar eventos               │
  └─────────────────────────────────────────────────────┘

POR QUÉ FUNCIONA:
  - El display NUNCA espera a la inferencia → video siempre fluido
  - El tracker recibe frames cada N*33ms (regular) → matching estable
  - model.track(persist=True) no necesita frame numbers consecutivos,
    solo necesita que entre llamada y llamada no haya pasado mucho
    tiempo visual. Con N=3 a 30fps → 100ms → persona se mueve ~15px
    → IoU sigue alto → track se mantiene.

POR QUÉ v3 FALLABA:
  - En v3, el frame se copiaba SOLO si el thread estaba libre
    (self._infer_done.is_set()). Si la inferencia tardaba mucho,
    se salteaban frames de forma IRREGULAR: el tracker veía
    frame 2, 8, 11, 19... con gaps impredecibles.
  - Ahora el main thread copia SIEMPRE cada N frames a un slot.
    Si el inference thread está ocupado, el slot se sobreescribe
    con el frame más reciente. El tracker puede perder algún frame
    pero el siguiente que procesa sigue siendo temporalmente cercano.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Deque, Tuple

import cv2
import numpy as np
from loguru import logger

from app.domain.events import EventPayload, TrackEvent
from app.domain.models import Detection, TrackEventType
from app.core.clock import now_iso
from app.core.tracking import TrackStateManager
from app.core.rules import RuleEngine, IntrusionRule


TRACK_COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 128),
    (128, 255, 0), (255, 255, 0), (0, 128, 255), (255, 0, 255),
    (0, 255, 128), (128, 0, 255),
]
ZONE_COLOR_DEFAULT = (0, 255, 255)
ZONE_COLOR_ACTIVE = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)


def _track_color(tid: int) -> Tuple[int, int, int]:
    return TRACK_COLORS[tid % len(TRACK_COLORS)]


class Pipeline:
    def __init__(
        self,
        site_id, camera_id, source, inferencer, zone_index,
        track_manager=None, rule_engine=None, intrusion_rule=None,
        snapshot_writer=None, event_store=None, notifier=None,
        processing_cfg=None, ui_cfg=None,
    ):
        self.site_id = site_id
        self.camera_id = camera_id
        self.source = source
        self.inferencer = inferencer
        self.zone_index = zone_index
        self.track_manager = track_manager
        self.rule_engine = rule_engine
        self.tracking_enabled = track_manager is not None
        self.intrusion_rule = intrusion_rule
        self.snapshot_writer = snapshot_writer
        self.event_store = event_store
        self.notifier = notifier
        self.processing_cfg = processing_cfg
        self.ui_cfg = ui_cfg
        self._trail_length = 30
        self._trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self._trail_length)
        )

        # ── Shared state (thread-safe) ──
        self._lock = threading.Lock()
        self._latest_dets: List[Detection] = []
        self._pending_events: List[Tuple[TrackEvent, np.ndarray, int]] = []
        self._infer_ms: float = 0.0
        self._infer_fps: float = 0.0

        # Slot para pasar frames al inference thread
        self._slot_frame: Optional[np.ndarray] = None
        self._slot_fidx: int = 0
        self._slot_new = threading.Event()      # señal: "hay frame nuevo en el slot"

        self._stop = threading.Event()

    @staticmethod
    def _to_dict(cfg):
        if cfg is None: return {}
        if isinstance(cfg, dict): return cfg
        if hasattr(cfg, "model_dump"): return cfg.model_dump()
        return dict(cfg)

    def run(self):
        proc = self._to_dict(self.processing_cfg)
        ui = self._to_dict(self.ui_cfg)

        resize_width = proc.get("resize_width", 960)
        show_window = bool(ui.get("show_window", True))
        window_name = str(ui.get("window_name", "MVP Monitor"))
        show_track_id = bool(ui.get("show_track_id", True))
        show_track_trail = bool(ui.get("show_track_trail", True))
        self._trail_length = int(ui.get("trail_length", 30))
        infer_every_n = max(1, int(ui.get("infer_every_n_frames", 3)))

        try:
            src_fps = float(self.source.fps())
        except Exception:
            src_fps = 25.0
        if src_fps <= 0:
            src_fps = 25.0
        frame_period = 1.0 / src_fps

        zone_ids = self.zone_index.zone_ids
        tracker_input_fps = src_fps / infer_every_n
        tracker_gap_ms = (infer_every_n / src_fps) * 1000

        logger.info(
            f"Pipeline start site={self.site_id} cam={self.camera_id} "
            f"tracking={'ON' if self.tracking_enabled else 'OFF'} "
            f"zones={zone_ids} src_fps={src_fps:.1f} "
            f"infer_every={infer_every_n} → tracker gets {tracker_input_fps:.1f} img/s "
            f"(gap {tracker_gap_ms:.0f}ms between frames)"
        )

        # Arrancar inference thread
        infer_thread = threading.Thread(
            target=self._inference_loop,
            args=(zone_ids,),
            daemon=True,
        )
        infer_thread.start()

        frame_idx = 0
        start_time = time.perf_counter()
        cleanup_counter = 0

        try:
            for ok, frame in self.source.frames():
                if not ok or self._stop.is_set():
                    if not self._stop.is_set():
                        logger.info("Fuente finalizada.")
                    break

                frame_idx += 1

                # Resize
                if resize_width:
                    h, w = frame.shape[:2]
                    if w != resize_width:
                        scale = resize_width / float(w)
                        frame = cv2.resize(frame, (resize_width, int(h * scale)),
                                           interpolation=cv2.INTER_LINEAR)

                # ── Cada N frames: copiar al slot para inference thread ──
                # SIEMPRE se copia (sobreescribe si el thread no procesó aún).
                # Así el tracker siempre recibe el frame más reciente cuando
                # está listo, manteniendo cadencia temporal regular.
                if frame_idx % infer_every_n == 0:
                    self._slot_frame = frame.copy()
                    self._slot_fidx = frame_idx
                    self._slot_new.set()

                # ── Procesar eventos pendientes del inference thread ──
                with self._lock:
                    pending = list(self._pending_events)
                    self._pending_events.clear()
                for ev, ev_frame, ev_fidx in pending:
                    self._handle_event(ev, ev_frame, ev_fidx)

                # ── Leer detecciones actuales ──
                with self._lock:
                    dets = list(self._latest_dets)
                    infer_ms = self._infer_ms
                    infer_fps = self._infer_fps

                # ── Display (SIEMPRE → fluido a fps nativo) ──
                if show_window:
                    vis = self._draw_overlay(
                        frame, dets, zone_ids,
                        show_track_id=show_track_id,
                        show_trail=show_track_trail,
                        infer_ms=infer_ms,
                        tracker_fps=infer_fps,
                        frame_idx=frame_idx,
                    )
                    cv2.imshow(window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Salida solicitada (q).")
                        self._stop.set()
                        break

                # ── Pacing: mantener velocidad real del video ──
                elapsed = time.perf_counter() - start_time
                expected = frame_idx * frame_period
                sleep_s = expected - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

                # Limpieza periódica
                cleanup_counter += 1
                if self.tracking_enabled and cleanup_counter >= 500:
                    with self._lock:
                        cleaned = self.track_manager.cleanup_old_tracks()
                    if cleaned > 0:
                        logger.debug(f"Cleaned {cleaned} old tracks")
                    cleanup_counter = 0

        finally:
            self._stop.set()
            self._slot_new.set()    # desbloquear thread
            infer_thread.join(timeout=3.0)
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            if self.tracking_enabled:
                stats = self.track_manager.get_stats()
                logger.info(f"Pipeline stopped. Tracking stats: {stats}")

    # ─── Inference thread ────────────────────────────────────────────

    def _inference_loop(self, zone_ids):
        """
        Thread de inferencia.
        
        Espera señal de frame nuevo → procesa → publica resultados.
        model.track(persist=True) mantiene estado entre llamadas.
        """
        logger.debug("Inference thread started")
        count = 0
        t_start = time.perf_counter()

        while not self._stop.is_set():
            # Esperar a que haya un frame en el slot
            got_signal = self._slot_new.wait(timeout=1.0)
            if self._stop.is_set():
                break
            if not got_signal:
                continue

            # Tomar el frame del slot
            self._slot_new.clear()
            frame = self._slot_frame
            fidx = self._slot_fidx

            if frame is None:
                continue

            # ── YOLO + Tracker ──
            t0 = time.perf_counter()
            try:
                dets = self.inferencer.infer(frame)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                dets = []
            ms = (time.perf_counter() - t0) * 1000

            count += 1
            elapsed = time.perf_counter() - t_start
            fps = count / elapsed if elapsed > 0 else 0

            # ── Zonas / Reglas ──
            events: List[TrackEvent] = []
            if self.tracking_enabled:
                events = self._process_tracking(dets, zone_ids)
            else:
                events = self._process_legacy(dets, zone_ids, fidx)

            # ── Publicar (thread-safe) ──
            with self._lock:
                self._latest_dets = dets
                self._infer_ms = ms
                self._infer_fps = fps
                for ev in events:
                    self._pending_events.append((ev, frame, fidx))

        logger.debug("Inference thread stopped")

    # ─── Processing ──────────────────────────────────────────────────

    def _process_tracking(self, dets, zone_ids):
        raw = self.track_manager.update(
            detections=dets, zone_index=self.zone_index, zone_ids=zone_ids)
        return self.rule_engine.process_events(raw) if self.rule_engine else raw

    def _process_legacy(self, dets, zone_ids, frame_idx):
        events = []
        if not self.intrusion_rule:
            return events
        zid = self.intrusion_rule.zone_id
        confs = [d.confidence for d in dets if self.zone_index.in_zone(zid, d.bbox.as_tuple())]
        if self.intrusion_rule.update(len(confs) > 0):
            events.append(TrackEvent(
                event_type=TrackEventType.INTRUSION, zone_id=zid, track_id=-1,
                confidence=max(confs) if confs else 0.0,
                meta={"detections": len(dets), "frame": frame_idx, "source": "legacy"}))
        return events

    # ─── Events ──────────────────────────────────────────────────────

    def _handle_event(self, ev, frame, frame_idx):
        ts = now_iso()
        snapshot_path = None
        if self.snapshot_writer:
            safe_ts = ts.replace(":", "-")
            tid_str = f"t{ev.track_id}" if ev.track_id >= 0 else "notk"
            fname = f"{self.site_id}_{self.camera_id}_{ev.zone_id}_{tid_str}_{safe_ts}.jpg"
            try:
                snapshot_path = self.snapshot_writer.write(frame, fname)
            except Exception as e:
                logger.warning(f"Snapshot error: {e}")

        payload = EventPayload(
            site_id=self.site_id, camera_id=self.camera_id, ts=ts,
            event_type=ev.event_type, zone_id=ev.zone_id,
            confidence=ev.confidence,
            track_id=ev.track_id if ev.track_id >= 0 else None,
            snapshot_path=snapshot_path,
            meta={**ev.meta, "dwell_seconds": ev.dwell_seconds, "frame": frame_idx},
        ).model_dump()

        if self.event_store:
            self.event_store.append(payload)
        logger.info(
            f"EVENT {ev.event_type} zone={ev.zone_id} "
            f"track={ev.track_id} conf={ev.confidence:.2f} "
            f"dwell={ev.dwell_seconds:.1f}s")
        if self.notifier:
            try:
                self.notifier.send(payload)
            except Exception as e:
                logger.warning(f"Webhook error: {e}")

    # ─── Visualization ───────────────────────────────────────────────

    def _draw_overlay(self, frame, dets, zone_ids,
                      show_track_id=True, show_trail=True,
                      infer_ms=0.0, tracker_fps=0.0, frame_idx=0):
        vis = frame.copy()

        zones_active = set()
        if self.tracking_enabled:
            for z in zone_ids:
                if self.track_manager.tracks_in_zone(z):
                    zones_active.add(z)
        else:
            for d in dets:
                for z in zone_ids:
                    if self.zone_index.in_zone(z, d.bbox.as_tuple()):
                        zones_active.add(z)

        for zid in zone_ids:
            pts = self.zone_index.get_polygon_points(zid)
            if not pts: continue
            act = zid in zones_active
            col = ZONE_COLOR_ACTIVE if act else ZONE_COLOR_DEFAULT
            np_pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [np_pts], True, col, 3 if act else 2)
            if act:
                ov = vis.copy()
                cv2.fillPoly(ov, [np_pts], col)
                cv2.addWeighted(ov, 0.12, vis, 0.88, 0, vis)
            cv2.putText(vis, zid, (pts[0][0], pts[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        for d in dets:
            b = d.bbox
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
            cx, cy = int(b.center[0]), int(b.center[1])
            col = _track_color(d.track_id) if d.track_id is not None else (255, 255, 255)
            in_z = any(self.zone_index.in_zone(z, b.as_tuple()) for z in zone_ids)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.circle(vis, (cx, cy), 4, col, -1)
            parts = []
            if show_track_id and d.track_id is not None:
                parts.append(f"ID:{d.track_id}")
            parts.append(f"{d.confidence:.2f}")
            if in_z: parts.append("IN")
            label = " ".join(parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), col, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            if show_trail and d.track_id is not None:
                self._trails[d.track_id].append((cx, cy))
                trail = list(self._trails[d.track_id])
                for i in range(1, len(trail)):
                    cv2.line(vis, trail[i-1], trail[i], col, max(1, int(2*i/len(trail))))

        lines = [
            f"SITE={self.site_id} CAM={self.camera_id}",
            f"Frame: {frame_idx}  Infer: {infer_ms:.0f}ms ({tracker_fps:.1f} tracker fps)",
        ]
        if self.tracking_enabled:
            st = self.track_manager.get_stats()
            lines.append(f"Active: {st['active_tracks']} | Total: {st['total_tracks_seen']}")
            for z in zone_ids:
                iz = len(self.track_manager.tracks_in_zone(z))
                uq = self.track_manager.unique_count_for_zone(z)
                lines.append(f"{z}: {iz} in | {uq} unique")

        lh = 22
        ph = len(lines) * lh + 10
        ov = vis.copy()
        cv2.rectangle(ov, (5, 5), (450, ph), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.6, vis, 0.4, 0, vis)
        for i, ln in enumerate(lines):
            cv2.putText(vis, ln, (10, 22 + i*lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
        return vis
