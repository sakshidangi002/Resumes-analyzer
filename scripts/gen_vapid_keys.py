"""Generate a VAPID keypair for Web Push and print the values you need to
paste into your `.env`.

Run once on the server (or any machine — keys are not server-specific):

    python scripts/gen_vapid_keys.py

Then add the printed lines to `Attendance Management/backend/.env`. Restart
the FastAPI app afterwards.

Re-running this script generates a NEW keypair and INVALIDATES every existing
push subscription (browsers verify pushes against the public key they
subscribed with). Only regenerate if a key was compromised.
"""
from __future__ import annotations

import base64
import sys


def _to_urlsafe_b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def main() -> int:
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError:
        print(
            "ERROR: `cryptography` is required.\n"
            "  pip install cryptography\n",
            file=sys.stderr,
        )
        return 1

    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())

    # Private key — PKCS8 DER, then urlsafe base64 (pywebpush accepts this).
    priv_der = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_b64 = _to_urlsafe_b64(priv_der)

    # Public key — raw uncompressed point (65 bytes: 0x04 || X || Y).
    public_numbers = private_key.public_key().public_numbers()
    x = public_numbers.x.to_bytes(32, "big")
    y = public_numbers.y.to_bytes(32, "big")
    raw_public = b"\x04" + x + y
    public_b64 = _to_urlsafe_b64(raw_public)

    print()
    print("=" * 72)
    print(" VAPID keypair generated. Paste these into your backend .env:")
    print("=" * 72)
    print(f'VAPID_PUBLIC_KEY={public_b64}')
    print(f'VAPID_PRIVATE_KEY={private_b64}')
    print(f'VAPID_CLAIM_EMAIL=mailto:your-admin@your-company.com')
    print("=" * 72)
    print()
    print("Restart the FastAPI app so the new keys take effect.")
    print(
        "NOTE: If you previously had subscriptions, they are now invalid — "
        "each user's browser will re-subscribe automatically on their next visit."
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
