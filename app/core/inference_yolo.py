from __future__ import annotations
from typing import List, Dict, Any
from ultralytics import YOLO
import numpy as np

class YoloPersonInferencer:
    def __init__(self, model_path: str, conf: float, iou: float, classes: List[int], imgsz: int):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.imgsz = imgsz

    def infer(self, frame_bgr) -> List[Dict[str, Any]]:
        # Ultralytics espera numpy BGR OK
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False
        )
        dets: List[Dict[str, Any]] = []
        r = results[0]
        if r.boxes is None:
            return dets

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)

        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            dets.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(c),
            })
        return dets
