# core/inference_yolo.py
"""
Inferencia YOLO con tracking nativo (BoT-SORT / ByteTrack).

model.track(persist=True) mantiene estado entre llamadas consecutivas.
No importa el número de frame — lo que importa es que entre llamada y
llamada el objeto no se haya movido demasiado (IoU > ~0.3).

Con skip=2 a 30fps → ~66ms entre llamadas → movimiento ~10-20px → OK.
Con skip=3 a 30fps → ~100ms entre llamadas → movimiento ~20-40px → OK.
"""
from __future__ import annotations

import tempfile
import os
from typing import List, Optional
from pathlib import Path

import numpy as np
from loguru import logger

from app.domain.models import Detection, BBox


class YoloInferencer:
    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        iou: float = 0.50,
        classes: Optional[List[int]] = None,
        imgsz: int = 640,
        tracking_enabled: bool = True,
        tracker_type: str = "botsort",
        tracker_config: Optional[dict] = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes or [0]
        self.imgsz = imgsz
        self.tracking_enabled = tracking_enabled
        self.tracker_type = tracker_type
        self._tracker_yaml: Optional[str] = None

        if tracker_config and tracking_enabled:
            self._tracker_yaml = self._build_tracker_yaml(tracker_type, tracker_config)

        logger.info(
            f"YoloInferencer: model={model_path} tracking={tracking_enabled} "
            f"tracker={tracker_type} conf={conf} iou={iou} classes={self.classes}"
        )

    @staticmethod
    def _build_tracker_yaml(tracker_type: str, config: dict) -> str:
        """
        Lee el YAML default de Ultralytics y sobreescribe nuestros valores.
        Así nunca falta un campo (como fuse_score).
        """
        import yaml

        base = {}
        try:
            import ultralytics
            cfg_dir = Path(ultralytics.__file__).parent / "cfg" / "trackers"
            default_yaml = cfg_dir / f"{tracker_type}.yaml"
            if default_yaml.exists():
                with open(default_yaml, "r") as f:
                    base = yaml.safe_load(f) or {}
                logger.debug(f"Loaded Ultralytics defaults from {default_yaml}")
        except Exception as e:
            logger.warning(f"Could not load Ultralytics defaults: {e}")

        if not base:
            if tracker_type == "botsort":
                base = {
                    "tracker_type": "botsort",
                    "track_high_thresh": 0.3, "track_low_thresh": 0.1,
                    "new_track_thresh": 0.4, "track_buffer": 60,
                    "match_thresh": 0.8, "fuse_score": True,
                    "gmc_method": "sparseOptFlow",
                    "proximity_thresh": 0.5, "appearance_thresh": 0.25,
                    "with_reid": False,
                }
            else:
                base = {
                    "tracker_type": "bytetrack",
                    "track_high_thresh": 0.3, "track_low_thresh": 0.1,
                    "new_track_thresh": 0.4, "track_buffer": 60,
                    "match_thresh": 0.8, "fuse_score": True,
                }

        base.update(config)

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{tracker_type}_", delete=False)
        yaml.safe_dump(base, tmp, default_flow_style=False)
        tmp.close()

        # ── Diagnóstico claro ──
        logger.info("=" * 60)
        logger.info("TRACKER CONFIG FINAL:")
        logger.info(f"  tracker_type    = {base.get('tracker_type')}")
        logger.info(f"  with_reid       = {base.get('with_reid')}")
        logger.info(f"  track_buffer    = {base.get('track_buffer')}")
        logger.info(f"  new_track_thresh= {base.get('new_track_thresh')}")
        logger.info(f"  match_thresh    = {base.get('match_thresh')}")
        logger.info(f"  appearance_thresh= {base.get('appearance_thresh')}")
        logger.info(f"  proximity_thresh = {base.get('proximity_thresh')}")
        if base.get("with_reid"):
            logger.info("  >>> ReID ACTIVO (primera vez descarga modelo ~5MB)")
        else:
            logger.warning("  >>> ReID INACTIVO — IDs se perderán al salir del cuadro")
        logger.info(f"  YAML path: {tmp.name}")
        logger.info("=" * 60)

        return tmp.name

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.tracking_enabled:
            return self._infer_track(frame_bgr)
        return self._infer_predict(frame_bgr)

    def _infer_track(self, frame_bgr: np.ndarray) -> List[Detection]:
        tracker_arg = self._tracker_yaml or f"{self.tracker_type}.yaml"
        results = self.model.track(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            persist=True,
            tracker=tracker_arg,
            verbose=False,
        )
        return self._parse(results, tracking=True)

    def _infer_predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr, conf=self.conf, iou=self.iou,
            classes=self.classes, imgsz=self.imgsz, verbose=False)
        return self._parse(results, tracking=False)

    @staticmethod
    def _parse(results, tracking: bool) -> List[Detection]:
        dets: List[Detection] = []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return dets
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
        track_ids = None
        if tracking and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy() if hasattr(boxes.id, "cpu") else np.array(boxes.id)
        for i, ((x1, y1, x2, y2), conf, cls_id) in enumerate(zip(xyxy, confs, cls_ids)):
            tid = int(track_ids[i]) if track_ids is not None else None
            dets.append(Detection(
                bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                confidence=float(conf), class_id=int(cls_id),
                class_name=r.names.get(int(cls_id), "unknown"), track_id=tid))
        return dets

    def reset_tracker(self):
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers = []

    def __del__(self):
        if self._tracker_yaml and os.path.exists(self._tracker_yaml):
            try: os.unlink(self._tracker_yaml)
            except OSError: pass
