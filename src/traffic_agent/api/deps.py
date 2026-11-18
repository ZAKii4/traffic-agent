"""Dépendances injectées dans les routes : paramètres, détecteur, utilisateur authentifié."""

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from traffic_agent.api.security import decode_access_token
from traffic_agent.settings import Settings, get_settings
from traffic_agent.vision.detector import DetectorLike

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_detector(request: Request) -> DetectorLike:
    detector: DetectorLike | None = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modèle non chargé")
    return detector


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)], settings: SettingsDep
) -> str:
    try:
        return decode_access_token(token, settings.jwt_secret.get_secret_value())
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Jeton invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


DetectorDep = Annotated[DetectorLike, Depends(get_detector)]
UserDep = Annotated[str, Depends(get_current_user)]
