"""Tests de l'API avec un faux détecteur : aucun modèle n'est chargé."""

from collections.abc import Iterator

import cv2
import numpy as np
import pytest
import supervision as sv
from fastapi.testclient import TestClient

from traffic_agent.api.deps import get_detector
from traffic_agent.api.main import app
from traffic_agent.settings import Settings, get_settings

PASSWORD = "mot-de-passe-de-test"
SECRET = "secret-de-test-suffisamment-long-pour-hs256"


class FakeDetector:
    names = {2: "car", 7: "truck", 0: "person"}

    def detect(self, frame: np.ndarray) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array([[10, 20, 110, 90], [200, 40, 380, 210], [5, 5, 15, 30]], dtype=float),
            confidence=np.array([0.91, 0.84, 0.77]),
            class_id=np.array([2, 7, 0]),
        )


def make_settings() -> Settings:
    return Settings(jwt_secret=SECRET, api_password=PASSWORD, load_model=False)


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = make_settings
    app.dependency_overrides[get_detector] = FakeDetector
    get_settings.cache_clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TA_JWT_SECRET", SECRET)
        mp.setenv("TA_API_PASSWORD", PASSWORD)
        mp.setenv("TA_LOAD_MODEL", "false")
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def token(client: TestClient, password: str = PASSWORD) -> str:
    r = client.post("/api/v1/auth/token", data={"username": "admin", "password": password})
    assert r.status_code == 200
    return str(r.json()["access_token"])


def png_bytes() -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}


def test_mauvais_mot_de_passe_refuse(client: TestClient) -> None:
    r = client.post("/api/v1/auth/token", data={"username": "admin", "password": "faux"})
    assert r.status_code == 401


def test_comptage_sans_jeton_refuse(client: TestClient) -> None:
    r = client.post("/api/v1/count/image", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 401


def test_jeton_invalide_refuse(client: TestClient) -> None:
    r = client.post(
        "/api/v1/count/image",
        files={"file": ("a.png", png_bytes(), "image/png")},
        headers={"Authorization": "Bearer faux.jeton.jwt"},
    )
    assert r.status_code == 401


def test_comptage_ne_garde_que_les_vehicules(client: TestClient) -> None:
    r = client.post(
        "/api/v1/count/image",
        files={"file": ("a.png", png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {token(client)}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["by_class"] == {"car": 1, "truck": 1}


def test_format_refuse(client: TestClient) -> None:
    r = client.post(
        "/api/v1/count/image",
        files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        headers={"Authorization": f"Bearer {token(client)}"},
    )
    assert r.status_code == 415


def test_image_illisible(client: TestClient) -> None:
    r = client.post(
        "/api/v1/count/image",
        files={"file": ("a.png", b"pas une image", "image/png")},
        headers={"Authorization": f"Bearer {token(client)}"},
    )
    assert r.status_code == 422


def test_image_trop_lourde(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        jwt_secret=SECRET, api_password=PASSWORD, load_model=False, max_image_mb=0.0001
    )
    r = client.post(
        "/api/v1/count/image",
        files={"file": ("a.png", png_bytes() * 10, "image/png")},
        headers={"Authorization": f"Bearer {token(client)}"},
    )
    assert r.status_code == 413


def test_secret_trop_court_refuse() -> None:
    with pytest.raises(ValueError, match="32"):
        Settings(jwt_secret="court", api_password=PASSWORD)
