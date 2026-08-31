from backend.services.resume_utils import (
    dump_text_list,
    normalize_email,
    normalize_name,
    normalize_phone,
    parse_text_list,
    sanitize_embedding_text,
)


def test_normalizers_are_stable_for_duplicate_detection():
    assert normalize_email("  Candidate@Example.COM ") == "candidate@example.com"
    assert normalize_phone("+91 (98765) 43210") == "9876543210"
    assert normalize_name("  Ada   Lovelace ") == "ada lovelace"


def test_text_lists_support_legacy_storage_formats():
    assert parse_text_list('["Python", "SQL"]') == ["Python", "SQL"]
    assert parse_text_list("Python, SQL\nDocker") == ["Python", "SQL", "Docker"]
    assert dump_text_list(["Python", "python", "SQL"]) == '["Python", "SQL"]'


def test_embedding_sanitizer_removes_symbol_noise():
    assert sanitize_embedding_text("Name\n\nSkills: Python\n!!!!!!!!!!!!!!!") == "Name Skills: Python"
