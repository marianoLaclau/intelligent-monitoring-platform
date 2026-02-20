# adapters/evidence/snapshot.py
from __future__ import annotations
import os
import cv2

class SnapshotWriter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def write(self, frame_bgr, filename: str) -> str:
        path = os.path.join(self.out_dir, filename)
        ok = cv2.imwrite(path, frame_bgr)
        if not ok:
            raise RuntimeError(f"No se pudo guardar snapshot: {path}")
        return path
