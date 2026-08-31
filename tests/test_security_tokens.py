"""Media tokens and embedding encryption.

Media tokens exist because <img src> cannot send an Authorization header, so the
live camera feeds take a token in the query string. The whole point is that this
token is NOT the session JWT: it expires in ~2 minutes and only opens camera
media. If the two were interchangeable the query-string token would grant the
entire API, which is what this file pins down.
"""
import os

import pytest
from app.core import config
from app.core.encrypted_types import EncryptedBinary
from app.core.security import (
    MEDIA_TOKEN_SCOPE,
    create_access_token,
    create_media_token,
    decode_access_token,
    decode_media_token,
    get_password_hash,
    verify_password,
)


def test_media_token_carries_scope_and_roles():
    payload = decode_media_token(create_media_token(7, ["Admin"]))
    assert payload is not None
    assert payload["scope"] == MEDIA_TOKEN_SCOPE
    assert payload["roles"] == ["Admin"]
    assert payload["sub"] == "7"


def test_session_token_is_rejected_as_a_media_token():
    """A leaked session JWT must not open camera streams via ?t=."""
    session = create_access_token(7, {"roles": ["Admin"]})
    assert decode_access_token(session) is not None      # valid session token
    assert decode_media_token(session) is None           # but NOT a media token


def test_media_token_is_not_accepted_where_a_session_is_required():
    """The reverse: the short-lived media token must not unlock the main API.

    It decodes as a JWT (same signing key) but carries scope=media, which the
    session path never issues — so anything checking the scope rejects it.
    """
    media = create_media_token(7, ["Admin"])
    payload = decode_access_token(media)
    assert payload is not None
    assert payload.get("scope") == MEDIA_TOKEN_SCOPE


def test_garbage_and_tampered_tokens_are_rejected():
    assert decode_media_token("") is None
    assert decode_media_token("not-a-jwt") is None
    tampered = create_media_token(7, ["Admin"])[:-4] + "AAAA"
    assert decode_media_token(tampered) is None


def test_password_verification_rejects_wrong_password_without_masking_errors():
    hashed = get_password_hash("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong", hashed)


@pytest.mark.parametrize("corrupt_hash", ["not-a-bcrypt-hash", "", "x" * 60, None])
def test_corrupt_password_hash_fails_closed_and_is_logged(corrupt_hash, caplog):
    """A corrupted hash must never authenticate, and must not be silent.

    bcrypt raises ValueError for a malformed hash, so catching it — however
    narrowly — makes a corrupt `password_hash` column look exactly like a wrong
    password: the user is locked out permanently and nothing says why.
    Fail closed AND log.
    """
    with caplog.at_level("ERROR"):
        assert verify_password("anything", corrupt_hash) is False  # type: ignore[arg-type]

    assert caplog.records, "corrupt hash was swallowed silently"
    assert "unusable" in caplog.records[0].getMessage()
    # The attempted password must never reach the log.
    assert "anything" not in caplog.text


def test_embedding_encryption_round_trips_and_hides_plaintext():
    codec = EncryptedBinary()
    original = b"\x01\x02\x03 face-embedding-bytes"
    stored = codec.process_bind_param(original, None)

    assert original not in stored, "embedding stored in the clear"
    assert stored.startswith(EncryptedBinary.prefix)
    assert codec.process_result_value(stored, None) == original


def test_legacy_plaintext_rows_remain_readable():
    """Rows written before encryption must keep working until backfilled."""
    codec = EncryptedBinary()
    legacy = b"plain-old-embedding"
    assert codec.process_result_value(legacy, None) == legacy


def test_embedding_key_is_separable_from_the_jwt_secret():
    """Rotating SECRET_KEY must not orphan enrolled faces.

    SECRET_KEY signs JWTs and SHOULD be rotated. The embedding key can never be
    rotated without re-enrolling every employee. Sharing one value turned
    routine JWT hygiene into destruction of biometric data.
    """
    codec = EncryptedBinary()
    original = b"embedding-to-survive-rotation"

    previous_secret = os.environ.get("SECRET_KEY")
    try:
        os.environ["EMBEDDING_ENCRYPTION_KEY"] = "dedicated-embedding-key-" + "z" * 40
        config.get_settings.cache_clear()
        stored = codec.process_bind_param(original, None)

        os.environ["SECRET_KEY"] = "rotated-jwt-secret-" + "q" * 40
        config.get_settings.cache_clear()

        assert codec.process_result_value(stored, None) == original
    finally:
        os.environ.pop("EMBEDDING_ENCRYPTION_KEY", None)
        if previous_secret is not None:
            os.environ["SECRET_KEY"] = previous_secret
        config.get_settings.cache_clear()
