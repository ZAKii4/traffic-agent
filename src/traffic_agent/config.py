"""Chargement et validation de la configuration des caméras (configs/cameras.yaml)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from traffic_agent.vision.counter import Line


class LineConfig(BaseModel):
    """Ligne de comptage en coordonnées relatives (0 = bord gauche/haut, 1 = bord droit/bas)."""

    id: str
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def non_degeneree(self) -> "LineConfig":
        if (self.x1, self.y1) == (self.x2, self.y2):
            raise ValueError(f"la ligne '{self.id}' a ses deux extrémités confondues")
        return self

    def to_pixels(self, width: int, height: int) -> Line:
        return Line(self.x1 * width, self.y1 * height, self.x2 * width, self.y2 * height)


class CameraConfig(BaseModel):
    id: str
    description: str = ""
    stride: int = Field(default=1, ge=1)
    classes: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 7])
    lines: list[LineConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiants_uniques(self) -> "CameraConfig":
        ids = [line.id for line in self.lines]
        if len(ids) != len(set(ids)):
            raise ValueError(f"la caméra '{self.id}' a des identifiants de ligne en double")
        return self


class AppConfig(BaseModel):
    cameras: list[CameraConfig] = Field(min_length=1)

    def camera(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        known = ", ".join(c.id for c in self.cameras)
        raise KeyError(f"caméra inconnue '{camera_id}' (disponibles : {known})")


def load_config(path: str | Path = "configs/cameras.yaml") -> AppConfig:
    """Lit le fichier YAML et lève une erreur claire s'il est invalide."""
    with Path(path).open(encoding="utf-8") as f:
        return AppConfig.model_validate(yaml.safe_load(f))
