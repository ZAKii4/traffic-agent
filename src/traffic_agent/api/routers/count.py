from collections import Counter

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from traffic_agent.api.deps import DetectorDep, SettingsDep, UserDep
from traffic_agent.api.schemas import CountResponse, Detection

router = APIRouter(prefix="/api/v1/count", tags=["comptage"])

ALLOWED_TYPES = {"image/jpeg", "image/png"}


@router.post(
    "/image",
    response_model=CountResponse,
    summary="Compter les véhicules sur une image",
    responses={401: {"description": "Non authentifié"}, 413: {}, 415: {}, 422: {}},
)
async def count_image(
    file: UploadFile, detector: DetectorDep, settings: SettingsDep, user: UserDep
) -> CountResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Formats acceptés : JPEG, PNG")
    max_bytes = int(settings.max_image_mb * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"Image trop lourde ({settings.max_image_mb:g} Mo maximum)",
        )
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Image illisible")

    detections = await run_in_threadpool(detector.detect, frame)
    detections = detections[np.isin(detections.class_id, settings.vehicle_classes)]
    items = [
        Detection(
            class_name=detector.names[int(class_id)],
            confidence=round(float(conf), 3),
            box=(float(x1), float(y1), float(x2), float(y2)),
        )
        for (x1, y1, x2, y2), conf, class_id in zip(
            detections.xyxy, detections.confidence, detections.class_id, strict=True
        )
    ]
    by_class = Counter(item.class_name for item in items)
    return CountResponse(total=len(items), by_class=dict(by_class), detections=items)
