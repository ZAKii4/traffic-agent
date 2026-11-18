"""Pipeline complet (détection simulée, vrai tracker, vrai compteur), sans modèle ni vidéo."""

import numpy as np
import supervision as sv

from traffic_agent.config import CameraConfig
from traffic_agent.vision.pipeline import count_frames

WIDTH, HEIGHT, FPS = 640, 480, 25.0


class FakeDetector:
    """Renvoie à chaque appel la prochaine liste de boîtes prévue par le scénario."""

    names = {2: "car", 7: "truck"}

    def __init__(self, script: list[list[tuple[float, float, float, float, int]]]) -> None:
        self.script = iter(script)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        boxes = next(self.script)
        if not boxes:
            return sv.Detections.empty()
        xyxy = np.array([b[:4] for b in boxes], dtype=float)
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.full(len(boxes), 0.9),
            class_id=np.array([b[4] for b in boxes]),
        )


def camera() -> CameraConfig:
    return CameraConfig.model_validate(
        {"id": "test", "lines": [{"id": "milieu", "x1": 0, "y1": 0.5, "x2": 1, "y2": 0.5}]}
    )


def frames(n: int) -> list[np.ndarray]:
    return [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)] * n


def descending_car(n: int, class_id: int = 2) -> list[list[tuple[float, float, float, float, int]]]:
    """Une voiture qui descend du haut vers le bas de l'image en n images."""
    return [[(300, 40 + 12 * i, 360, 80 + 12 * i, class_id)] for i in range(n)]


def test_une_voiture_qui_traverse_est_comptee_une_fois() -> None:
    events = list(
        count_frames(frames(30), FPS, WIDTH, HEIGHT, camera(), FakeDetector(descending_car(30)))
    )
    assert [(e.line_id, e.direction, e.class_name) for e in events] == [("milieu", "in", "car")]


def test_les_classes_hors_configuration_sont_ignorees() -> None:
    script = descending_car(30, class_id=99)
    assert list(count_frames(frames(30), FPS, WIDTH, HEIGHT, camera(), FakeDetector(script))) == []


def test_images_sans_detection() -> None:
    script: list[list[tuple[float, float, float, float, int]]] = [[] for _ in range(10)]
    assert list(count_frames(frames(10), FPS, WIDTH, HEIGHT, camera(), FakeDetector(script))) == []
