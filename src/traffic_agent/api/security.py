"""Hachage des mots de passe et jetons JWT."""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

ALGORITHM = "HS256"


def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def verify_password(password: str, password_hash: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash)


def create_access_token(subject: str, secret: str, expire_minutes: int) -> str:
    now = datetime.now(UTC)
    payload = {"sub": subject, "iat": now, "exp": now + timedelta(minutes=expire_minutes)}
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> str:
    """Renvoie l'utilisateur du jeton ; lève jwt.InvalidTokenError s'il est invalide ou expiré."""
    payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
    return str(payload["sub"])
