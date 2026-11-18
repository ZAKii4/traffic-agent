"""Relie détection, suivi et comptage, et produit un flux d'événements de franchissement."""

from collections.abc import Iterable, Iterator

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from traffic_agent.config import CameraConfig
from traffic_agent.vision.counter import CrossingEvent, LineCounter
from traffic_agent.vision.detector import DetectorLike


def count_frames(
    frames: Iterable[np.ndarray],
    fps: float,
    width: int,
    height: int,
    camera: CameraConfig,
    detector: DetectorLike,
) -> Iterator[CrossingEvent]:
    """Compte les franchissements sur une suite d'images (déjà échantillonnée selon le stride)."""
    tracker = ByteTrackTracker()
    counters = [
        LineCounter(camera.id, line.id, line.to_pixels(width, height)) for line in camera.lines
    ]
    for i, frame in enumerate(frames):
        detections = detector.detect(frame)
        detections = detections[np.isin(detections.class_id, camera.classes)]
        detections = tracker.update(detections)
        if detections.tracker_id is None or len(detections) == 0:
            continue
        detections = detections[detections.tracker_id != -1]
        anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        video_time_s = i * camera.stride / fps
        for (x, y), track_id, class_id in zip(
            anchors, detections.tracker_id, detections.class_id, strict=True
        ):
            for counter in counters:
                event = counter.update(
                    int(track_id), detector.names[int(class_id)], float(x), float(y), video_time_s
                )
                if event is not None:
                    yield event


def count_video(
    video_path: str, camera: CameraConfig, detector: DetectorLike
) -> Iterator[CrossingEvent]:
    """Compte les franchissements dans un fichier vidéo."""
    info = sv.VideoInfo.from_video_path(video_path)
    frames = sv.get_video_frames_generator(video_path, stride=camera.stride)
    yield from count_frames(frames, info.fps, info.width, info.height, camera, detector)
