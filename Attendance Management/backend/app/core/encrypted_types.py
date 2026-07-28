"""Transparent application-level encryption for biometric embeddings."""

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.types import LargeBinary, TypeDecorator

from app.core.config import get_settings


class EncryptedBinary(TypeDecorator):
    impl = LargeBinary
    cache_ok = True
    prefix = b"SWENC1:"

    @staticmethod
    def _cipher() -> Fernet:
        settings = get_settings()
        # Prefer the dedicated key; fall back to secret_key so installations
        # that predate EMBEDDING_ENCRYPTION_KEY still decrypt their existing
        # rows. See config.py for why these must be separable: rotating
        # secret_key is routine, rotating this one orphans every stored
        # embedding.
        secret = (settings.embedding_encryption_key or settings.secret_key).encode("utf-8")
        key = base64.urlsafe_b64encode(hashlib.sha256(secret).digest())
        return Fernet(key)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        raw = bytes(value)
        return self.prefix + self._cipher().encrypt(raw)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        raw = bytes(value)
        if not raw.startswith(self.prefix):
            # Legacy rows remain readable so they can be re-saved encrypted.
            return raw
        try:
            return self._cipher().decrypt(raw[len(self.prefix):])
        except InvalidToken as exc:
            raise ValueError("Embedding could not be decrypted with the current key") from exc
