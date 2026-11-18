"""Fine couche autour d'Ultralytics : le reste du code ne voit que des sv.Detections."""

from typing import Protocol

import numpy as np
import supervision as sv


class DetectorLike(Protocol):
    """Ce que le pipeline attend d'un détecteur (remplaçable, ou simulé en test)."""

    names: dict[int, str]

    def detect(self, frame: np.ndarray) -> sv.Detections: ...


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloDetector:
    def __init__(
        self,
        weights: str = "yolo26n.pt",
        imgsz: int = 1280,
        conf: float = 0.25,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device or best_device()
        self.names: dict[int, str] = dict(self.model.names)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        return sv.Detections.from_ultralytics(result)
