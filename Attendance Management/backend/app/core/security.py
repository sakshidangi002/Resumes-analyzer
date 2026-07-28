import bcrypt
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from jose import jwt
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a password against its stored bcrypt hash.

    Fails CLOSED: any problem returns False, so a broken hash can never
    authenticate anyone. But it also LOGS, which is the part that was missing.

    bcrypt raises ValueError for a malformed or truncated hash, so simply
    catching it (however narrowly) makes a corrupted `password_hash` column
    indistinguishable from a wrong password: the user is locked out
    permanently, every attempt looks like a normal failed login, and nothing
    anywhere says why. Narrowing the except clause did not change that on its
    own — the corrupt-hash case IS a ValueError. Surfacing it does.
    """
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except (ValueError, TypeError, AttributeError):
        # No password material in the log — the stored hash is unusable, and
        # that is a data-integrity problem for an operator to investigate.
        logger.error(
            "Stored password hash is unusable (malformed or wrong type); "
            "treating verification as failed. This account cannot log in until "
            "its password is reset.",
            exc_info=True,
        )
        return False


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def create_access_token(
    subject: str | int,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode = {"exp": expire, "sub": str(subject)}
    if extra_claims:
        to_encode.update(extra_claims)
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> Optional[dict[str, Any]]:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Media (camera stream) tokens
# ---------------------------------------------------------------------------
# Live MJPEG feeds and JPEG previews are rendered by <img src="..."> tags, which
# cannot send an Authorization header -- which is why those endpoints were left
# unauthenticated. Instead the UI calls an authenticated endpoint to mint a
# SHORT-LIVED, NARROWLY-SCOPED token and puts that in the query string.
#
# Why a separate short-lived media token rather than reusing the session JWT:
# is valid for 30 days and grants the whole API. It ends up in browser history,
# proxy logs and Referer headers. This one is worthless after a couple of
# minutes and only opens camera media.
#
# LIMITATION, by design: an MJPEG response never ends, so the token is checked
# when the connection is ESTABLISHED, not continuously. A stream opened one
# second before expiry keeps running until the client disconnects. Shrinking the
# TTL does not change that -- terminating live viewers would need a revocation
# check inside the frame loop.

MEDIA_TOKEN_SCOPE = "media"
MEDIA_TOKEN_TTL_SECONDS = 120


def create_media_token(subject: str | int, roles: list[str]) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(seconds=MEDIA_TOKEN_TTL_SECONDS)
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "scope": MEDIA_TOKEN_SCOPE,
        "roles": list(roles),
    }
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)


def decode_media_token(token: str) -> Optional[dict[str, Any]]:
    """Decode and verify a media token. Returns None unless the signature is
    valid, it has not expired, AND it carries the media scope -- so a full
    session JWT cannot be replayed here, and a media token cannot be replayed
    against the main API (which never looks for this scope)."""
    payload = decode_access_token(token)
    if not payload or payload.get("scope") != MEDIA_TOKEN_SCOPE:
        return None
    return payload
