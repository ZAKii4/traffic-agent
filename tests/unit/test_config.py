from pathlib import Path

import pytest
from pydantic import ValidationError

from traffic_agent.config import AppConfig, LineConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def camera_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "cam",
        "lines": [{"id": "l1", "x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5}],
    }
    return {**base, **overrides}


def test_le_fichier_du_depot_est_valide() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "cameras.yaml")
    camera = config.camera("demo-autoroute")
    assert [line.id for line in camera.lines] == ["vers_camera", "s_eloigne"]


def test_conversion_en_pixels() -> None:
    line = LineConfig(id="l", x1=0.0, y1=0.75, x2=0.5, y2=0.75).to_pixels(3840, 2160)
    assert (line.x1, line.y1, line.x2, line.y2) == (0.0, 1620.0, 1920.0, 1620.0)


def test_coordonnee_hors_limites_refusee() -> None:
    with pytest.raises(ValidationError):
        LineConfig(id="l", x1=0.0, y1=1.2, x2=1.0, y2=0.5)


def test_ligne_degeneree_refusee() -> None:
    with pytest.raises(ValidationError, match="confondues"):
        LineConfig(id="l", x1=0.3, y1=0.3, x2=0.3, y2=0.3)


def test_identifiants_de_ligne_en_double_refuses() -> None:
    line = {"id": "l1", "x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5}
    with pytest.raises(ValidationError, match="double"):
        AppConfig.model_validate({"cameras": [camera_dict(lines=[line, line])]})


def test_camera_sans_ligne_refusee() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"cameras": [camera_dict(lines=[])]})


def test_camera_inconnue() -> None:
    config = AppConfig.model_validate({"cameras": [camera_dict()]})
    with pytest.raises(KeyError, match="inconnue"):
        config.camera("autre")
