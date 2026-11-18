"""Schémas d'entrée et de sortie de l'API (ils apparaissent dans la documentation Swagger)."""

from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class Detection(BaseModel):
    class_name: str
    confidence: float = Field(ge=0, le=1)
    box: tuple[float, float, float, float] = Field(description="x1, y1, x2, y2 en pixels")


class CountResponse(BaseModel):
    total: int
    by_class: dict[str, int]
    detections: list[Detection]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "total": 2,
                    "by_class": {"car": 1, "truck": 1},
                    "detections": [
                        {"class_name": "car", "confidence": 0.91, "box": [10, 20, 110, 90]},
                        {"class_name": "truck", "confidence": 0.84, "box": [200, 40, 380, 210]},
                    ],
                }
            ]
        }
    }


class Health(BaseModel):
    status: str
    model_loaded: bool
