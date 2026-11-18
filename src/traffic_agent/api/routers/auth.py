from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from traffic_agent.api.deps import SettingsDep
from traffic_agent.api.schemas import Token
from traffic_agent.api.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["authentification"])


@lru_cache
def _password_hash(password: str) -> bytes:
    return hash_password(password)


@router.post("/token", response_model=Token, summary="Obtenir un jeton d'accès")
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], settings: SettingsDep) -> Token:
    expected_hash = _password_hash(settings.api_password.get_secret_value())
    if form.username != settings.api_username or not verify_password(
        form.password, expected_hash
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Identifiants incorrects")
    token = create_access_token(
        form.username, settings.jwt_secret.get_secret_value(), settings.jwt_expire_minutes
    )
    return Token(access_token=token)
