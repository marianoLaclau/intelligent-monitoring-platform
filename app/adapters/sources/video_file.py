from __future__ import annotations
import os
import cv2
from typing import Iterator, Tuple, Optional


class VideoFileSource:
    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        if not os.path.exists(self.path):
            raise RuntimeError(f"Video no encontrado: {self.path} (cwd={os.getcwd()})")

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.path}")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def fps(self) -> float:
        if self.cap is None:
            self.open()
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 0 else 25.0  # fallback razonable

    def frames(self) -> Iterator[Tuple[bool, any]]:
        if self.cap is None:
            self.open()

        while True:
            ok, frame = self.cap.read()
            if ok:
                yield True, frame
                continue

            # EOF
            if not self.loop:
                yield False, None
                return

            # loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
