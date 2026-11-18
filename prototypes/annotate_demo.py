"""Video annotee : boites, identifiants, trajectoires et lignes de comptage (config commune)."""

import numpy as np
import supervision as sv
from demo_config import IMGSZ, LINES, VEHICLE_CLASSES
from supervision.assets import VideoAssets, download_assets
from trackers import ByteTrackTracker
from ultralytics import YOLO

video_path = download_assets(VideoAssets.VEHICLES)
info = sv.VideoInfo.from_video_path(video_path)
W, H = info.width, info.height
zones = {
    n: sv.LineZone(
        start=sv.Point(a * W, b * H),
        end=sv.Point(c * W, d * H),
        triggering_anchors=(sv.Position.BOTTOM_CENTER,),
    )
    for n, (a, b, c, d) in LINES.items()
}
model = YOLO("yolo26n.pt")
tracker = ByteTrackTracker()
box = sv.BoxAnnotator(thickness=4)
label = sv.LabelAnnotator(text_scale=1.5, text_thickness=3)
trace = sv.TraceAnnotator(thickness=4, trace_length=60)
line_annotator = sv.LineZoneAnnotator(thickness=4, text_scale=2)


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    result = model(frame, device="mps", imgsz=IMGSZ, verbose=False)[0]
    det = sv.Detections.from_ultralytics(result)
    det = det[np.isin(det.class_id, VEHICLE_CLASSES)]
    det = tracker.update(det)
    det = det[det.tracker_id != -1]
    for zone in zones.values():
        zone.trigger(det)
    labels = [f"#{t} {model.names[c]}" for t, c in zip(det.tracker_id, det.class_id, strict=True)]
    frame = trace.annotate(frame.copy(), det)
    frame = box.annotate(frame, det)
    frame = label.annotate(frame, det, labels)
    for zone in zones.values():
        frame = line_annotator.annotate(frame, zone)
    return frame


sv.process_video(video_path, "runs/annotated_raw.mp4", callback)
for n, z in zones.items():
    print(n, "| in:", z.in_count, "| out:", z.out_count)
