"""Compte les franchissements avec sv.LineZone ET avec notre propre compteur, puis compare."""

import numpy as np
import supervision as sv
from supervision.assets import VideoAssets, download_assets
from trackers import ByteTrackTracker
from ultralytics import YOLO

from demo_config import GROUND_TRUTH, IMGSZ, LINES, VEHICLE_CLASSES


class LineCounter:
    """Notre compteur : point bas-centre, segment borné, un comptage par véhicule."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.last_side: dict[int, float] = {}
        self.counted: set[int] = set()

    def side(self, x: float, y: float) -> float:
        return (self.x2 - self.x1) * (y - self.y1) - (self.y2 - self.y1) * (x - self.x1)

    def within(self, x: float, y: float) -> bool:
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        t = ((x - self.x1) * dx + (y - self.y1) * dy) / (dx * dx + dy * dy)
        return 0.0 <= t <= 1.0

    def update(self, tid: int, x: float, y: float) -> bool:
        s = self.side(x, y)
        prev = self.last_side.get(tid)
        self.last_side[tid] = s
        if prev is None or tid in self.counted or not self.within(x, y) or prev * s >= 0:
            return False
        self.counted.add(tid)
        return True


video_path = download_assets(VideoAssets.VEHICLES)
info = sv.VideoInfo.from_video_path(video_path)
W, H = info.width, info.height

zones, ours = {}, {}
for name, (a, b, c, d) in LINES.items():
    zones[name] = sv.LineZone(
        start=sv.Point(a * W, b * H),
        end=sv.Point(c * W, d * H),
        triggering_anchors=(sv.Position.BOTTOM_CENTER,),
    )
    ours[name] = LineCounter(a * W, b * H, c * W, d * H)

model = YOLO("yolo26n.pt")
tracker = ByteTrackTracker()
sv_counts = {n: 0 for n in LINES}
events = []

for i, frame in enumerate(sv.get_video_frames_generator(video_path)):
    result = model(frame, device="mps", imgsz=IMGSZ, verbose=False)[0]
    det = sv.Detections.from_ultralytics(result)
    det = det[np.isin(det.class_id, VEHICLE_CLASSES)]
    det = tracker.update(det)
    det = det[det.tracker_id != -1]
    anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    t = i / info.fps
    for name in LINES:
        crossed_in, crossed_out = zones[name].trigger(det)
        sv_counts[name] += int(crossed_in.sum() + crossed_out.sum())
        for (x, y), tid, cid in zip(anchors, det.tracker_id, det.class_id):
            if ours[name].update(int(tid), float(x), float(y)):
                events.append((t, name, int(tid), model.names[int(cid)]))

print("\nFranchissements détectés par notre compteur :")
for t, name, tid, cls in events:
    print(f"  {t:5.1f} s  {name:12s}  #{tid:<3d} {cls}")

print("\nLigne          vérité  sv.LineZone  notre compteur")
for name in LINES:
    n_ours = sum(1 for e in events if e[1] == name)
    print(f"{name:14s} {GROUND_TRUTH[name]:6d}  {sv_counts[name]:11d}  {n_ours:14d}")
