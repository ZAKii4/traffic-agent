"""Application FastAPI : création, chargement du modèle au démarrage, routes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from traffic_agent.api.routers import auth, count
from traffic_agent.api.schemas import Health
from traffic_agent.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.detector = None
    if settings.load_model:
        from traffic_agent.vision.detector import YoloDetector

        app.state.detector = YoloDetector(weights=settings.model_weights)
    yield


app = FastAPI(
    title="Traffic Agent API",
    version="0.1.0",
    description="Comptage de véhicules par vision : détection, suivi et comptage par classe.",
    lifespan=lifespan,
)
app.include_router(auth.router)
app.include_router(count.router)


@app.get("/health", response_model=Health, tags=["supervision"], summary="Le service répond")
def health(request: Request) -> Health:
    return Health(status="ok", model_loaded=request.app.state.detector is not None)
