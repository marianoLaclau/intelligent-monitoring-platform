from __future__ import annotations
import time
import cv2
from loguru import logger
from app.domain.events import EventPayload
from app.core.clock import now_iso


class Pipeline:
    def __init__(
        self,
        site_id: str,
        camera_id: str,
        source,
        inferencer,
        zone_index,
        intrusion_rule,
        snapshot_writer,
        event_store,
        notifier=None,
        processing_cfg=None,
        ui_cfg=None,
    ):
        self.site_id = site_id
        self.camera_id = camera_id
        self.source = source
        self.inferencer = inferencer
        self.zone_index = zone_index
        self.intrusion_rule = intrusion_rule
        self.snapshot_writer = snapshot_writer
        self.event_store = event_store
        self.notifier = notifier
        self.processing_cfg = processing_cfg
        self.ui_cfg = ui_cfg

    @staticmethod
    def _to_dict(cfg) -> dict:
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return cfg
        if hasattr(cfg, "model_dump"):  # pydantic v2
            return cfg.model_dump()
        return dict(cfg)

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def run(self) -> None:
        proc = self._to_dict(self.processing_cfg)
        ui = self._to_dict(self.ui_cfg)

        resize_width = proc.get("resize_width", 960)

        # UI
        show_window = bool(ui.get("show_window", True))
        window_name = str(ui.get("window_name", "MVP Monitor"))
        wait_ms_override = ui.get("wait_ms", None)  # si lo seteás, pisa el realtime

        # Inference sampling (para performance sin romper “realtime”)
        infer_every_n = max(1, int(ui.get("infer_every_n_frames", 2)))

        # Realtime: respetar FPS del video (si el source lo expone)
        try:
            src_fps = float(self.source.fps())
        except Exception:
            src_fps = 25.0
        if src_fps <= 0:
            src_fps = 25.0

        frame_period = 1.0 / src_fps

        logger.info(
            f"Pipeline start site={self.site_id} cam={self.camera_id} "
            f"realtime_fps={src_fps:.2f} infer_every_n={infer_every_n} resize_width={resize_width}"
        )

        zone_id = self.intrusion_rule.zone_id
        frame_idx = 0
        last_dets = []

        # Para timing realtime
        start_monotonic = time.perf_counter()
        shown_frames = 0

        try:
            for ok, frame in self.source.frames():
                if not ok:
                    logger.info("Fuente finalizada.")
                    break

                frame_idx += 1

                # Resize
                if resize_width:
                    h, w = frame.shape[:2]
                    if w != resize_width:
                        scale = resize_width / float(w)
                        frame = cv2.resize(
                            frame,
                            (resize_width, int(h * scale)),
                            interpolation=cv2.INTER_LINEAR,
                        )

                # Inferencia cada N frames (sin acelerar ni frenar el video)
                if frame_idx % infer_every_n == 0:
                    last_dets = self.inferencer.infer(frame)

                dets = last_dets

                # Cálculo intrusión por punto de referencia (centro bbox)
                in_zone_points = []
                for d in dets:
                    bbox = d["bbox"]
                    cx, cy = self._bbox_center(bbox)
                    inside = self.zone_index.in_zone(zone_id, bbox)  # usa centro internamente
                    in_zone_points.append((cx, cy, inside, float(d["conf"]), bbox))

                in_zone_confs = [conf for (_cx, _cy, inside, conf, _bbox) in in_zone_points if inside]
                is_in_zone = len(in_zone_confs) > 0
                fired = self.intrusion_rule.update(is_in_zone)

                # ---------- UI overlay ----------
                if show_window:
                    vis = frame.copy()

                    # Zona (polígono)
                    poly = self.zone_index.polys.get(zone_id)
                    if poly is not None:
                        pts = [(int(x), int(y)) for x, y in list(poly.exterior.coords)]
                        for i in range(len(pts) - 1):
                            cv2.line(vis, pts[i], pts[i + 1], (0, 255, 255), 2)

                    # Detecciones: bbox + punto referencia + label inside
                    for (cx, cy, inside, conf, bbox) in in_zone_points:
                        x1, y1, x2, y2 = map(int, bbox)
                        color = (0, 255, 0) if inside else (255, 255, 255)

                        # bbox
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                        # punto referencia (centro)
                        cv2.circle(vis, (cx, cy), 4, color, -1)

                        # texto de conf + inside
                        txt = f"{conf:.2f} {'IN' if inside else 'OUT'}"
                        cv2.putText(
                            vis,
                            txt,
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                    # Labels globales (siempre visibles)
                    cv2.putText(
                        vis,
                        f"SITE={self.site_id} CAM={self.camera_id}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        vis,
                        f"INTRUSION_ZONE: {'YES' if is_in_zone else 'NO'}",
                        (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0) if is_in_zone else (255, 255, 255),
                        2,
                    )

                    

                    

                    cv2.imshow(window_name, vis)

                    # Control de salida
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Salida solicitada (q).")
                        break

                # ---------- Realtime pacing ----------
                # Mostramos el frame N en el instante: start + N * period
                shown_frames += 1
                target_time = start_monotonic + (shown_frames * frame_period)
                now = time.perf_counter()
                sleep_s = target_time - now

                # Si wait_ms_override está seteado, NO hacemos pacing realtime (solo waitKey)
                if wait_ms_override is None and sleep_s > 0:
                    time.sleep(sleep_s)
                # ------------------------------------

                # evento (snapshot + jsonl + webhook)
                if fired:
                    ts = now_iso()
                    best_conf = float(max(in_zone_confs)) if in_zone_confs else 0.0

                    snapshot_path = None
                    if self.snapshot_writer:
                        safe_ts = ts.replace(":", "-")
                        fname = f"{self.site_id}_{self.camera_id}_{zone_id}_{safe_ts}.jpg"
                        snapshot_path = self.snapshot_writer.write(frame, fname)

                    payload = EventPayload(
                        site_id=self.site_id,
                        camera_id=self.camera_id,
                        ts=ts,
                        event_type="intrusion",
                        zone_id=zone_id,
                        confidence=best_conf,
                        snapshot_path=snapshot_path,
                        meta={"detections": len(dets), "frame": frame_idx},
                    ).model_dump()

                    self.event_store.append(payload)
                    logger.info(
                        f"EVENT intrusion zone={zone_id} conf={best_conf:.2f} snapshot={snapshot_path}"
                    )

                    if self.notifier:
                        try:
                            code = self.notifier.send(payload)
                            logger.info(f"Webhook status={code}")
                        except Exception as e:
                            logger.warning(f"Webhook error: {e}")

        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass



