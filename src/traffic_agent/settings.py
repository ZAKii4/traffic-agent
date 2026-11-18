"""Paramètres de l'application, lus dans les variables d'environnement (préfixe TA_) ou .env."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TA_", env_file=".env", extra="ignore")

    jwt_secret: SecretStr
    jwt_expire_minutes: int = Field(default=30, ge=1)
    api_username: str = "admin"
    api_password: SecretStr
    model_weights: str = "yolo26n.pt"
    load_model: bool = True
    max_image_mb: float = Field(default=5, gt=0)
    vehicle_classes: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 7])

    @field_validator("jwt_secret")
    @classmethod
    def secret_assez_long(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("TA_JWT_SECRET doit faire au moins 32 caractères")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
