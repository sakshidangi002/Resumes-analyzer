"""Masking for government IDs and bank account numbers.

Client-side masking is cosmetic: the full value still travels in the JSON and is
readable in the browser's Network tab, in logs, and by anyone calling the API
directly. Masking therefore happens HERE, on the way out of the API, and the
unmasked value is served only by an explicit, role-gated, audited reveal
endpoint.
"""
from typing import Optional

# U+2022 BULLET. Kept as a module constant because `is_masked` must recognise
# exactly what `mask_secret` produces.
MASK_CHAR = "•"
MASK_PREFIX = MASK_CHAR * 4


def mask_secret(value: Optional[str], keep: int = 4) -> Optional[str]:
    """Return `value` with everything but the last `keep` characters hidden.

    None stays None (the field is genuinely unset) rather than becoming a mask,
    so the UI can still tell "not provided" from "hidden".
    """
    if value is None:
        return None
    compact = "".join(value.split())
    if not compact:
        return value
    if len(compact) <= keep:
        # Too short to reveal any tail without disclosing the whole value.
        return MASK_PREFIX
    return f"{MASK_PREFIX} {compact[-keep:]}"


def is_masked(value: Optional[str]) -> bool:
    """True if `value` looks like something we emitted from `mask_secret`.

    Write paths use this to DROP masked input. The edit form is populated from a
    masked GET, so a save that passed the value straight through would overwrite
    the real Aadhaar/PAN/account number in the database with its own mask.
    """
    return bool(value) and MASK_CHAR in value
