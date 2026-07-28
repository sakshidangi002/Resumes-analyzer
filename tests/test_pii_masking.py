"""Employee PII masking.

Regression cover for two real bugs:
  * masking was client-side only, so the full Aadhaar/PAN still travelled in the
    JSON and was readable in the browser's Network tab;
  * once masking moved server-side, the edit form was seeded from a MASKED GET,
    so an untouched save wrote "•••• 1234" over the real stored value.
"""
from app.core.pii import MASK_PREFIX, is_masked, mask_secret


def test_mask_reveals_only_the_last_four_characters():
    assert mask_secret("123456789012") == f"{MASK_PREFIX} 9012"
    assert mask_secret("ABCDE1234F") == f"{MASK_PREFIX} 234F"


def test_mask_ignores_internal_whitespace_so_grouping_does_not_leak_more():
    # "1234 5678 9012" must not expose "9012" plus formatting hints.
    assert mask_secret("1234 5678 9012") == mask_secret("123456789012")


def test_short_values_are_fully_hidden():
    # Revealing the last 4 of a 4-character value would disclose all of it.
    assert mask_secret("1234") == MASK_PREFIX
    assert mask_secret("12") == MASK_PREFIX


def test_absent_values_stay_absent():
    # None means "not provided" and must stay distinguishable from "hidden".
    assert mask_secret(None) is None
    assert mask_secret("") == ""


def test_is_masked_detects_our_own_output_only():
    assert is_masked(mask_secret("123456789012"))
    assert is_masked(mask_secret("1234"))
    # A real identifier must never be mistaken for a mask, or a genuine edit
    # would be silently discarded by the write guard.
    assert not is_masked("123456789012")
    assert not is_masked("ABCDE1234F")
    assert not is_masked(None)
    assert not is_masked("")


def test_masked_value_never_contains_the_original():
    secret = "987654321098"
    masked = mask_secret(secret)
    assert secret not in masked
    assert masked.endswith("1098")
