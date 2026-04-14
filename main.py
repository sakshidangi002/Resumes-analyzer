import json
import logging
import re
from datetime import datetime, timedelta
import os

import pdfplumber
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction model (NuExtract-1.5-smol) – lazy singleton
# ---------------------------------------------------------------------------
EXTRACT_MODEL = "numind/NuExtract-1.5-smol"
_extract_tokenizer = None
_extract_model = None


def _get_extract_tokenizer():
    global _extract_tokenizer
    if _extract_tokenizer is None:
        from transformers import AutoTokenizer
        logger.info("Loading NuExtract tokenizer…")
        _extract_tokenizer = AutoTokenizer.from_pretrained(EXTRACT_MODEL, use_fast=False)
        logger.info("NuExtract tokenizer ready.")
    return _extract_tokenizer


def _get_extract_model():
    global _extract_model
    if _extract_model is None:
        from transformers import AutoModelForCausalLM
        logger.info("Loading NuExtract model…")
        _extract_model = AutoModelForCausalLM.from_pretrained(EXTRACT_MODEL)
        logger.info("NuExtract model ready.")
    return _extract_model


def preload_extract_model() -> None:
    """Load extraction tokenizer and model so first upload is not slow (cold start)."""
    _get_extract_tokenizer()
    _get_extract_model()


# ---------------------------------------------------------------------------
# Chat model – lazy singleton

import os as _os

CHAT_MODEL = _os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
_chat_pipe = None


def _get_chat_pipe():
    global _chat_pipe
    if _chat_pipe is None:
        from transformers import pipeline
        logger.info("Loading chat pipeline: %s …", CHAT_MODEL)
        _chat_pipe = pipeline(
            "text-generation",
            model=CHAT_MODEL,
            do_sample=False,
        )
        logger.info("Chat pipeline ready: %s", CHAT_MODEL)
    return _chat_pipe


def preload_chat_model() -> None:
    """Load the chat model at startup so the first chat request is fast. Call from API lifespan if PRELOAD_CHAT_MODEL=1."""
    _get_chat_pipe()


# ---------------------------------------------------------------------------
# File text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    """
    Extract plain text from PDF.
    Default uses pdfplumber (better layout but slower). You can opt into
    a faster extractor by setting PDF_EXTRACTOR=pypdf.
    """
    def _extract_with_pypdf() -> str:
        from pypdf import PdfReader

        reader = PdfReader(path)
        out: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                out.append(t)
        return "\n".join(out).strip()

    def _extract_with_pdfplumber() -> str:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()

    extractor = (os.getenv("PDF_EXTRACTOR", "") or "").strip().lower()
    preferred = "pypdf" if extractor in {"pypdf", "pdf"} else "pdfplumber"

    t1 = ""
    t2 = ""
    try:
        t1 = _extract_with_pypdf() if preferred == "pypdf" else _extract_with_pdfplumber()
    except Exception:
        t1 = ""
    try:
        t2 = _extract_with_pdfplumber() if preferred == "pypdf" else _extract_with_pypdf()
    except Exception:
        t2 = ""

    if not t2:
        return (t1 or "").strip()
    if not t1:
        return (t2 or "").strip()

    def _format_score(txt: str) -> float:
        """
        Prefer outputs that preserve section structure (line breaks) and contain
        expected headers. This avoids pypdf-style collapsed lines polluting skills.
        """
        if not txt:
            return 0.0
        c, meta = _compute_extraction_confidence(txt)
        # Reward line structure + header presence
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        line_bonus = min(0.15, len(lines) / 200.0)
        header_bonus = 0.0
        low = txt.lower()
        if "skills" in low:
            header_bonus += 0.08
        if "experience" in low:
            header_bonus += 0.05
        if "projects" in low:
            header_bonus += 0.03
        # Penalize very long lines (collapsed formatting)
        long_lines = sum(1 for ln in lines[:80] if len(ln) > 140)
        collapse_penalty = min(0.20, long_lines * 0.03)
        return float(c) + line_bonus + header_bonus - collapse_penalty

    # Normalize both candidates before choosing so downstream parsing sees stable text.
    n1, meta1 = normalize_resume_text(t1)
    n2, meta2 = normalize_resume_text(t2)

    # Choose the extraction with higher confidence (better line structure, less junk).
    # Score using normalized text (more representative of parsing inputs).
    s1 = _format_score(n1)
    s2 = _format_score(n2)
    if s2 > s1 + 0.03:
        return n2.strip()
    if s1 > s2 + 0.03:
        return n1.strip()
    # Otherwise prefer the longer normalized text (more content)
    return (n1 if len(n1) >= len(n2) else n2).strip()


def extract_text_from_docx(path: str) -> str:
    """Extract plain text from a DOCX file."""
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ---------------------------------------------------------------------------
# Text quality / sanitization (internal reliability helpers)
# ---------------------------------------------------------------------------

def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _strip_footnote_numbers(token: str) -> str:
    """
    Remove trailing PDF artifacts like '127)' / '(12)' / '12' that often appear
    after wrapped bullets or footnotes.
    """
    if not token:
        return ""
    s = str(token).strip()
    s = re.sub(r"\s*\(?\b\d{1,4}\)?\s*$", "", s).strip()
    return s


def _symbol_ratio(s: str) -> float:
    if not s:
        return 0.0
    total = len(s)
    if total <= 0:
        return 0.0
    sym = sum(1 for ch in s if not (ch.isalnum() or ch.isspace()))
    return sym / total


def _drop_symbol_heavy_lines(text: str, *, max_ratio: float = 0.38) -> str:
    """
    Remove lines that are mostly symbols/noise (common in poorly extracted PDFs).
    Keeps content lines and returns joined text.
    """
    if not text:
        return ""
    out: list[str] = []
    for ln in (text or "").splitlines():
        raw = ln.strip()
        if not raw:
            continue
        if _symbol_ratio(raw) > max_ratio and not re.search(r"[A-Za-z0-9]", raw):
            continue
        if _symbol_ratio(raw) > max_ratio and len(raw) > 12:
            continue
        out.append(raw)
    return "\n".join(out).strip()


def normalize_resume_text(text: str) -> tuple[str, dict]:
    """
    Normalize raw extracted resume text into a more consistent representation.
    Goal: reduce PDF layout variability (footers, hyphenation, column artifacts)
    so downstream parsing is less resume-specific.
    Returns (normalized_text, meta).
    """
    raw = (text or "").replace("\r", "\n")
    if not raw.strip():
        return "", {"actions": ["empty_input"]}

    meta: dict = {"actions": []}

    # 1) Trim extreme whitespace while preserving line breaks.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]

    # 2) Drop very symbol-heavy lines (common PDF noise).
    before = len(lines)
    lines = [ln for ln in lines if _symbol_ratio(ln) <= 0.55 or re.search(r"[A-Za-z0-9]", ln)]
    if len(lines) != before:
        meta["actions"].append("drop_symbol_heavy_lines")

    # 3) Remove repeated footer/header lines (e.g., "EDUCATION SKILLS EXPERIENCE" repeated every page).
    # Detect repeats by normalized form.
    normed = [_norm_header(ln) for ln in lines]
    counts: dict[str, int] = {}
    for n in normed:
        if not n:
            continue
        counts[n] = counts.get(n, 0) + 1

    # Treat as footer/header candidate if it repeats AND is short AND looks header-like.
    repeated = {
        n
        for n, c in counts.items()
        if c >= 3 and 3 <= len(n) <= 45 and _looks_like_header(n)
    }
    if repeated:
        before = len(lines)
        kept_lines: list[str] = []
        for ln in lines:
            n = _norm_header(ln)
            if n in repeated:
                continue
            kept_lines.append(ln)
        lines = kept_lines
        if len(lines) != before:
            meta["actions"].append("drop_repeated_footer_headers")

    # 4) De-hyphenate wrapped words: "develop-\nment" -> "development"
    # Only when a line ends with a hyphen and next line begins with a letter.
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.endswith("-") and i + 1 < len(lines) and re.match(r"^[A-Za-z]{2,}", lines[i + 1]):
            merged = (ln[:-1] + lines[i + 1]).strip()
            out.append(merged)
            meta["actions"].append("dehyphenate_wraps")
            i += 2
            continue
        out.append(ln)
        i += 1
    lines = out

    # 5) Limit pathological repeated characters (OCR/PDF glitches).
    before_join = "\n".join(lines)
    fixed = re.sub(r"(.)\1{9,}", r"\1\1\1", before_join)
    if fixed != before_join:
        meta["actions"].append("limit_repeats")

    normalized = fixed.strip()
    # Provide a few quality hints (used for downstream arbitration/fallbacks).
    conf, conf_meta = _compute_extraction_confidence(normalized)
    meta["confidence"] = conf
    meta["confidence_meta"] = conf_meta
    return normalized, meta


# Stopwords / generic noise for skill extraction
_SKILL_STOPWORDS = {"and", "or", "with", "using", "based", "of"}
_SKILL_GENERIC_WORDS = {
    "apps",
    "app",
    "system",
    "systems",
    "development",
    "software",
    "technology",
    "tools",
    "full",
    "stack",
    "developer",
    "developers",
    "management",
    "professional",
    "experience",
    "summary",
    "registration",
    "submission",
    "email",
    "phone",
    "address",
    "personal",
    "information",
    "objective",
    "career",
    "goals",
    "goal",
    "formulate",
    "execute",
    "strategies",
    "effective",
}
_SKILL_ADJECTIVES = {"scalable", "dynamic", "efficient", "robust", "reliable", "high", "low", "fast", "secure"}
_SKILL_VAGUE_PHRASES = {
    "dynamic content rendering",
    "api consumption",
}


def _fix_merged_words(text: str) -> str:
    """
    Best-effort fix for merged words like 'separationofconcerns' by inserting
    spaces around common glue tokens when embedded.
    """
    if not text:
        return ""
    t = str(text)
    # Split camelCase everywhere (safe)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)

    # Glue-token split ONLY for long no-space runs to avoid breaking real words
    # like "core" (c + or + e).
    def _fix_word(w: str) -> str:
        if not w:
            return w
        if " " in w:
            return w
        if len(w) < 12:
            return w
        # Only split glue tokens when there are at least 2 letters on each side
        for glue in ("of", "and", "with", "for", "to", "in"):
            w = re.sub(rf"(?i)([a-z]{{2,}})({glue})([a-z]{{2,}})", r"\1 \2 \3", w)
        return w

    parts = re.split(r"(\s+)", t)  # keep whitespace separators
    parts = [_fix_word(p) if not p.isspace() else p for p in parts]
    return "".join(parts)


def _clean_text_for_skill_tokens(text: str) -> str:
    """
    Mandatory first layer for skill extraction:
    - lowercase
    - remove stopwords
    - remove special chars except valid tech tokens (keeps . + # -)
    - fix merged words
    - normalize whitespace
    """
    if not text:
        return ""
    t = _fix_merged_words(text).lower()
    # Process line-by-line to preserve newlines for tokenization.
    cleaned_lines: list[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        # Keep . + # / - for tech tokens like c++, c#, .net, ci/cd
        s = re.sub(r"[^a-z0-9.+#/\-\s]", " ", s)
        s = s.replace("ci cd", "ci/cd")
        # Remove stopwords as standalone words
        for sw in _SKILL_STOPWORDS:
            s = re.sub(rf"\b{re.escape(sw)}\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            cleaned_lines.append(s)
    return "\n".join(cleaned_lines).strip()


def _tokenize_skill_candidates(text: str) -> list[str]:
    """
    Tokenization:
    - split by commas, bullets, line breaks and common separators
    - extract 2–3 word noun-phrase-like chunks
    """
    if not text:
        return []
    raw = _clean_text_for_skill_tokens(text)
    # Primary split: commas, bullets, pipes, semicolons, newlines.
    # Do NOT split on '.' because it breaks tokens like ASP.NET / Node.js.
    # Sentence periods are handled later by token validation + evidence gating.
    parts = re.split(r"[,\n\r|;•\u2022]+", raw)
    tokens: list[str] = []
    for p in parts:
        s = _collapse_whitespace(p)
        if not s:
            continue
        # Secondary split on separators but keep dot/plus/hash within tokens
        for x in re.split(r"\s{2,}|(?<!ci)/(?!=cd)", s):
            x = _collapse_whitespace(x)
            if x:
                tokens.append(x)

    # NOTE: we do NOT generate arbitrary n-grams here because it creates noisy
    # phrases ("operations committed", etc.). Multi-word skills are handled via:
    # - explicit SKILLS section parsing
    # - known vocabulary matching (e.g., "asp.net core", "azure devops")

    # De-dupe while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        t = _collapse_whitespace(t)
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _normalize_skill_token(tok: str) -> str:
    """
    Canonical normalization with special handling for common stacks.
    """
    t = _strip_footnote_numbers(tok)
    t = _collapse_whitespace(t)
    if not t:
        return ""
    # Prefer existing canonicalise_skill mapping first
    canon = canonicalise_skill(t)
    if canon:
        t = canon
    # Consolidations
    low = _norm_header(t)
    if low in {"asp.net core mvc", "asp.net core", "aspnetcore", "aspnet core"}:
        return "ASP.NET Core"
    if low in {"ci", "cd", "ci/cd", "ci cd", "ci cd automation", "cicd"}:
        return "CI/CD"
    if low in {"azure devops ci", "azure devops", "azure devops cicd"}:
        return "Azure DevOps CI/CD"
    if low in {"html"}:
        return "HTML"
    if low in {"css"}:
        return "CSS"
    return t


def _strip_skill_decorators(raw: str) -> str:
    """Strip leading bullets / hashtags used in resume templates (e.g. '# Good Communication')."""
    s = (raw or "").strip()
    s = re.sub(r"^[#•\-\*\u2022\u00b7\u2013\u2014\s]+", "", s).strip()
    return s


def _skill_is_noise(tok: str) -> bool:
    if not tok:
        return True
    raw = _strip_skill_decorators(str(tok))
    if not raw:
        return True
    low = _norm_header(raw)
    if low in _SKILL_STOPWORDS:
        return True
    if low in _SKILL_GENERIC_WORDS:
        return True
    if low in _SKILL_ADJECTIVES:
        return True
    if low in _SKILL_VAGUE_PHRASES:
        return True
    # School / board lines mistaken for skills (avoid bare 'board' — matches 'keyboard')
    if re.search(
        r"\b(hp\s+board|hbse|cbse|icse|nios|pseb|mp\s+board|up\s+board|state\s+board|central\s+board)\b",
        low,
    ):
        return True
    return False


def _name_tokens_for_skill_exclusion(full_name: str) -> set[str]:
    """
    Parts of the candidate name (lowercased, len>=3) that must never appear as a lone 'skill'.
    Prevents headers / PDF lines like 'Gourav' from showing under Other Skills.
    """
    if not full_name:
        return set()
    low = full_name.strip().lower()
    if low in {"unknown", "unknown candidate", "candidate"}:
        return set()
    parts = re.split(r"[\s.'\-]+", full_name.strip())
    return {p.lower() for p in parts if len(p) >= 3}


def _skill_is_excluded_name_token(skill: str, name_tokens: set[str]) -> bool:
    """True when a single-word skill is exactly a given-name / surname token."""
    if not skill or not name_tokens:
        return False
    s = _strip_footnote_numbers(str(skill)).strip()
    if not s:
        return False
    if _is_core_tech_label(s):
        return False
    low = s.lower()
    if " " in low:
        return False
    return low in name_tokens


def _infer_primary_role_label(weight_by_skill: dict[str, int]) -> str:
    """
    Primary role identity label (not a tool list).
    """
    if not weight_by_skill:
        return "Full Stack Development"
    backend_terms = {
        ".net",
        "asp.net",
        "asp.net core",
        "c#",
        "sql",
        "sql server",
        "microservices",
        "rest api",
        "api",
        "docker",
        "kubernetes",
        "azure",
        "aws",
    }
    frontend_terms = {"react", "angular", "vue", "javascript", "typescript", "html", "css", "next.js"}

    b = 0
    f = 0
    has_dotnet = False
    for k, w in weight_by_skill.items():
        kk = _norm_header(k)
        if kk in {"asp.net", "asp.net core", ".net", "c#"}:
            has_dotnet = True
        if any(term == kk for term in backend_terms):
            b += max(0, w)
        if any(term == kk for term in frontend_terms):
            f += max(0, w)

    if b >= (f + 4) and has_dotnet:
        return "Cloud-Native .NET Backend Engineering"
    if f >= (b + 4):
        return "Frontend Development"
    return "Full Stack Development"


def _compute_extraction_confidence(resume_text: str) -> tuple[float, dict]:
    """
    Internal-only confidence score (0–1) for how reliable downstream extraction
    is likely to be, based on text quality and presence of key sections.
    """
    text = (resume_text or "").strip()
    if not text:
        return 0.0, {"reason": "empty"}

    # Length: very short text is almost always unreliable.
    n_chars = len(text)
    length_score = min(1.0, n_chars / 1800.0)  # ~1-2 pages of text

    # Junk: symbol-heavy content suggests OCR/parse noise.
    slice_text = text[:8000]
    non_alnum = sum(1 for ch in slice_text if not (ch.isalnum() or ch.isspace()))
    junk_ratio = non_alnum / max(1, len(slice_text))
    junk_score = max(0.0, 1.0 - (junk_ratio * 2.2))  # >0.30 junk pushes low

    # Repetition: many duplicated lines indicates extraction loops / headers repeated.
    lines = [re.sub(r"\s+", " ", ln.strip().lower()) for ln in text.splitlines() if ln.strip()]
    uniq = set(lines)
    rep_ratio = 1.0 - (len(uniq) / max(1, len(lines)))
    rep_score = max(0.0, 1.0 - (rep_ratio * 1.8))

    # Sections: missing EXPERIENCE and SKILLS reduces confidence for evaluation.
    has_skills = any(_is_skills_header_line(ln) for ln in (resume_text or "").splitlines())
    has_exp = any(
        _norm_header(ln) in {
            "experience",
            "work experience",
            "work experiences",
            "work history",
            "professional experience",
            "employment history",
        }
        for ln in (resume_text or "").splitlines()
    )
    section_score = 1.0
    if not has_skills:
        section_score -= 0.25
    if not has_exp:
        section_score -= 0.25
    section_score = max(0.0, min(1.0, section_score))

    # Weighted blend. Bias toward text quality (junk/repetition) over mere length.
    score = (
        (0.25 * length_score)
        + (0.35 * junk_score)
        + (0.20 * rep_score)
        + (0.20 * section_score)
    )
    score = max(0.0, min(1.0, score))
    meta = {
        "n_chars": n_chars,
        "junk_ratio": round(junk_ratio, 4),
        "rep_ratio": round(rep_ratio, 4),
        "has_skills_section": bool(has_skills),
        "has_experience_section": bool(has_exp),
    }
    return score, meta

# ---------------------------------------------------------------------------
# Skills parsing helpers (non-LLM, reliable for "SKILLS" sections)
# ---------------------------------------------------------------------------

_SKILLS_SECTION_HEADERS = {
    # Original 17 (keep all)
    "skills",
    "competencies",
    "tech skills",
    "technical skills",
    "key skills",
    "core competencies",
    "skill set",
    "skillset",
    "tech stack",
    "technology stack",
    "technologies",
    "tools",
    "tools & technologies",
    "tools and technologies",
    "expertise",
    "technical expertise",
    "areas of expertise",
    "stack",
    # NEW: Additional 15+ variations
    "technical proficiencies",
    "proficiencies",
    "programming languages",
    "development tools",
    "software & platforms",
    "technology focus",
    "technical focus",
    "specialized skills",
    "professional skills",
    "technical background",
    "development languages",
    "frameworks & libraries",
    "languages & frameworks",
    "technical capabilities",
    "capabilities",
    "technical focus areas",
}

_COMMON_SECTION_HEADERS = {
    "profile",
    "summary",
    "objective",
    "career objective",
    "career summary",
    "professional summary",
    "professional experience",
    "work experience",
    "experience",
    "education",
    "projects",
    "certifications",
    "achievements",
    "awards",
    "languages",
    "interests",
}

_CORE_TECH_PRIMARY = {
    # Languages / runtimes
    "python",
    "java",
    "javascript",
    "typescript",
    "c#",
    "c++",
    "php",
    "go",
    "golang",
    "ruby",
    # Web frameworks / stacks
    "react",
    "angular",
    "vue",
    "node",
    "node.js",
    "express",
    "next.js",
    "django",
    "flask",
    "spring",
    ".net",
    "dotnet",
    "asp.net",
    # Data / db / cloud
    "sql",
    "mysql",
    "postgres",
    "postgresql",
    "mongodb",
    "nosql",
    "redis",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "k8s",
}


def _normalise_skill_token(tok: str) -> str:
    """
    Normalise a single skill name for consistent comparison and display.
    - Strips whitespace and duplicate spaces
    - Keeps ALL‑CAPS tokens (e.g. SQL, AWS) as-is
    - Otherwise uses title case (e.g. 'asp.net mvc' → 'Asp.Net Mvc')
    """
    tok = re.sub(r"\s+", " ", (tok or "").strip())
    if not tok:
        return ""
    if tok.isupper() and len(tok) <= 15:
        return tok
    # Preserve common tech casing
    low = tok.lower()
    if low in {"html", "css"}:
        return low.upper()
    return tok.title()


_EDUCATION_SKILL_WORDS = {
    "bachelor",
    "bachelors",
    "master",
    "masters",
    "degree",
    "education",
    "studies",
    "diploma",
    "university",
    "college",
    "school",
    "institute",
    "academy",
    "faculty",
}

# Certification keywords (indicates skill/certification, not education)
_CERTIFICATION_KEYWORDS = {
    "certified",
    "certification",
    "credentials",
    "badge",
    "accredited",
    "associate",
    "professional",
    "expert",
    "practitioner",
    "specialist",
    "architect",
    "engineer",
}

# Regex patterns for well-known certifications
_CERTIFICATION_PATTERNS = [
    r"AWS\s+(Certified|Solution|Developer|SysOps|DevOps|Data)",
    r"Google\s+Cloud\s+(Certified|Associate|Professional)",
    r"Azure.*Certificate",
    r"(CKA|CKAD|PMP|CISSP|SCRUM|CPA|OSCP|RHCE|LPIC|GIAC|CISM)\b",
    r"Certified\s+(Kubernetes|AWS|Azure|Google|Microsoft|Oracle|Salesforce)\s+\w+",
    r"AWS\s+(Certified\s+)?(Solutions\s+Architect|Developer|SysOps|DevOps|Data Analytics|Cloud Practitioner)",
    r"Microsoft\s+Certified.*",
]


def _looks_like_certification(text: str) -> bool:
    """
    Detect if text describes a certification (skill) rather than education.
    Examples of certifications (return True):
      - "AWS Certified Solutions Architect"
      - "Kubernetes Administrator (CKA)"
      - "Google Cloud Professional"
    Examples of education (return False):
      - "Bachelor of Science"
      - "Master's Degree"
    """
    if not text:
        return False
    s = str(text).lower()

    # Check 1: Known certification keywords (exclude if also has education markers)
    has_cert_word = any(w in s for w in _CERTIFICATION_KEYWORDS)
    has_edu_word = any(
        w in s for w in ("degree", "bachelor", "diploma", "university", "college")
    )
    if has_cert_word and not has_edu_word:
        return True

    # Check 2: Known certification patterns
    for pattern in _CERTIFICATION_PATTERNS:
        if re.search(pattern, s, re.IGNORECASE):
            return True

    return False


def _is_percentage_metric(text: str) -> bool:
    """
    Detect if text is a percentage-based metric (e.g., 'Studies 60%', 'Management 75%').
    These are not skills; they're usually proficiency ratings or profile scores.

    Returns True = filter out (it's a metric, not a skill)
    Returns False = keep (it might be a skill)
    """
    if not text:
        return False
    s = str(text).strip()
    # Fast path: if there is no '%' at all, it's not a percentage metric
    if "%" not in s:
        return False
    # Pattern: any word/phrase that contains a numeric percentage
    # Examples: "Studies 60%", "Corporate Affairs 75 %", "React 90%"
    if re.search(r"\d{1,3}\s*%", s):
        return True
    return False


def _is_education_like_phrase(text: str) -> bool:
    """
    Heuristic: detect tokens that actually describe education, not skills.
    Used to keep degrees/universities out of the skills list.

    Returns True = filter out (it's education)
    Returns False = keep (it's not education)
    """
    if not text:
        return False
    s = str(text).lower()

    # CHECK 0: Is it a certification? (highest priority)
    # Certifications are skills, NOT education, so return False (don't filter)
    if _looks_like_certification(s):
        return False

    # CHECK 1: Simple substring matching on education keywords
    if any(w in s for w in _EDUCATION_SKILL_WORDS):
        return True

    # CHECK 2: Regex for degree abbreviations (includes B.Ed / M.Ed)
    if re.search(
        r"\b(b\.?\s*sc|m\.?\s*sc|b\.?\s*tech|m\.?\s*tech|b\.?\s*ed|m\.?\s*ed|bca|mca|bba|mba)\b",
        s,
    ):
        return True

    # CHECK 3: Degree marker + year range
    if re.search(r"\b(19|20)\d{2}\b", s) and any(
        w in s for w in ("bachelor", "master", "degree", "diploma")
    ):
        return True

    # CHECK 4: School / education board (HP Board, CBSE, etc.) — not technical skills
    if re.search(
        r"\b(hp\s+board|hbse|cbse|icse|nios|pseb|mp\s+board|up\s+board|state\s+board|central\s+board)\b",
        s,
    ):
        return True

    return False


def _skill_is_supported_by_text(skill: str, resume_text: str) -> bool:
    """
    Guardrail against hallucinated skills:
    keep a skill only if it appears in the resume text (case-insensitive),
    using word-boundary matching for short tokens.
    """
    if not skill or not resume_text:
        return False
    s = str(skill).strip()
    if not s:
        return False
    text = (resume_text or "").lower()
    # normalise punctuation/spaces in both for matching like "asp.net" / "asp net"
    norm_text = re.sub(r"[^a-z0-9]+", " ", text)
    norm_skill = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    if not norm_skill:
        return False
    # For short tokens like "git", require full word match
    if len(norm_skill) <= 4 and " " not in norm_skill:
        return re.search(rf"\b{re.escape(norm_skill)}\b", norm_text) is not None
    # For multi-word skills, require all words present
    parts = [p for p in norm_skill.split() if p]
    if not parts:
        return False
    return all(re.search(rf"\b{re.escape(p)}\b", norm_text) for p in parts)


def _is_core_tech_label(label: str) -> bool:
    """
    Heuristic for whether a label is a core technical skill suitable for
    the 'Primary Skills' column (React, .NET, Node.js, AWS, etc.).
    """
    if not label:
        return False
    s = str(label).lower()
    # Direct keyword match
    if any(kw in s for kw in _CORE_TECH_PRIMARY):
        return True
    # Short all‑caps technology acronyms (SQL, AWS, GCP, SAP, REST)
    if label.isupper() and 2 <= len(label) <= 5:
        return True
    # Contains common tech separators like '.' or '+'
    if any(ch in s for ch in (".", "+", "#")):
        return True
    return False


def classify_skills(
    skills: list[str] | None,
    key_skills: list[str] | None,
) -> tuple[list[str], list[str]]:
    """
    Classify skills into:
      - primary_skills: top 3 core technologies / domains
      - other_skills: all remaining unique skills

    Priority order:
      1) key_skills as extracted by the LLM (meant to be core tech)
      2) remaining skills from the broader skills list

    Duplicates are removed case‑insensitively and names are normalised.
    """

    def _clean_list(items: list[str] | None) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in items or []:
            norm = _normalise_skill_token(str(raw))
            if not norm:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(norm)
        return cleaned

    ks = _clean_list(key_skills or [])
    all_sk = _clean_list(skills or [])

    # Merge, giving priority to key_skills ordering
    merged: list[str] = []
    seen_merged: set[str] = set()
    for s in ks + all_sk:
        key = s.lower()
        if key in seen_merged:
            continue
        seen_merged.add(key)
        merged.append(s)

    # Prefer clearly technical labels for primary skills
    primary: list[str] = []
    for s in merged:
        if _is_core_tech_label(s):
            primary.append(s)
        if len(primary) == 3:
            break

    # If we still have fewer than 3, backfill with remaining items
    if len(primary) < 3:
        for s in merged:
            if s in primary:
                continue
            primary.append(s)
            if len(primary) == 3:
                break

    primary_set = {s.lower() for s in primary}
    other = [s for s in merged if s.lower() not in primary_set]

    return primary, other


def _boost_domain_skills(
    resume_text: str,
    skills: list[str],
    key_skills: list[str],
    current_role: str,
    experience_summary: str,
) -> tuple[list[str], list[str]]:
    """
    Heuristically boost domain-defining technologies (like .NET) so they
    appear at the front of the skills / key_skills lists.

    This focuses on the tech the candidate actually works with in roles
    and projects, not just any mentioned language.
    """

    blob = " ".join(
        [
            (resume_text or ""),
            (current_role or ""),
            (experience_summary or ""),
        ]
    ).lower()

    def ensure_front(lst: list[str], label: str):
        # Keep display label as-is; classification will normalise later
        if label in lst:
            lst.remove(label)
        lst.insert(0, label)

    # .NET stack (.NET / ASP.NET / C#)
    net_triggers = [".net", " dotnet", " asp.net", " asp .net", " c# ", " c sharp"]
    if any(t in blob for t in net_triggers):
        # Prefer canonical trio: .NET, ASP.NET, MVC as leading primary skills
        patterns = [
            (".NET", r"\.net"),
            ("ASP.NET", r"asp\.?\.?net"),
            ("MVC", r"\bmvc\b"),
        ]
        for label, pat in patterns:
            if re.search(pat, blob, flags=re.IGNORECASE):
                if label not in skills:
                    skills.insert(0, label)
                else:
                    ensure_front(skills, label)
                if label not in key_skills:
                    key_skills.insert(0, label)
                else:
                    ensure_front(key_skills, label)

    # Frontend React focus
    react_triggers = [" react ", " react.js", " reactjs", " next.js", " nextjs"]
    if any(t in blob for t in react_triggers):
        canonical = "React"
        if canonical not in skills:
            skills.insert(0, canonical)
        else:
            ensure_front(skills, canonical)
        if canonical not in key_skills:
            key_skills.insert(0, canonical)
        else:
            ensure_front(key_skills, canonical)

    return skills, key_skills


def _norm_header(line: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (line or "").lower()).strip()


def _looks_like_header(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    norm = _norm_header(s)
    if norm in _COMMON_SECTION_HEADERS or norm in _SKILLS_SECTION_HEADERS:
        return True
    # Heuristic: short uppercase line is usually a section title
    if s.isupper() and 3 <= len(s) <= 40 and not any(ch.isdigit() for ch in s):
        return True
    return False


_HEADER_BAD_NAMES = {
    "resume",
    "curriculum vitae",
    "cv",
    "profile",
    "summary",
    "professional summary",
    "software engineer",
    "senior software engineer",
    "full stack developer",
    "full-stack developer",
    "backend developer",
    "frontend developer",
    "data scientist",
    "machine learning engineer",
    "devops engineer",
    "business development",
    "business development executive",
    "business development manager",
    "sales executive",
    "sales manager",
    "inside sales",
    "account manager",
    "marketing manager",
}


def _strip_contact_noise(s: str) -> str:
    """Remove obvious contact tokens from a line (emails/phones/urls)."""
    if not s:
        return ""
    email_re = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    phone_re = r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){6,14}\d"
    s = re.sub(email_re, " ", s)
    s = re.sub(phone_re, " ", s)
    s = re.sub(r"(https?://\S+|www\.\S+)", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -•|,:\t")
    return re.sub(r"\s+", " ", s).strip()


def _is_plausible_person_name(candidate: str) -> bool:
    """Heuristic: 2-4 word alphabetic-ish, not a known header/title."""
    c = _strip_contact_noise(candidate)
    if not c:
        return False
    norm = _norm_header(c)
    if not norm or norm in _HEADER_BAD_NAMES:
        return False
    # Never treat section headers as person names (even if ALL-CAPS).
    if norm in _COMMON_SECTION_HEADERS or norm in _SKILLS_SECTION_HEADERS:
        return False
    # Common header-y phrases that show up as "names" in PDFs
    if any(x in norm for x in ("key skills", "technical skills", "programming languages", "languages")):
        return False
    if norm in {"phone", "email", "location", "experience"}:
        return False
    if "software development" in norm:
        return False
    # Reject obvious technical labels (skills/tools) that sometimes appear near headers.
    # IMPORTANT: do NOT use _is_core_tech_label here because it substring-matches
    # (e.g. "Gourav" contains "go"). Keep this check explicit and conservative.
    if re.search(
        r"\b(api|asp\.?net|\.net|mvc|sql|aws|gcp|azure|react|node\.?js|javascript|typescript|python|java|c\+\+|c#)\b",
        norm,
    ):
        return False
    # Reject lines that clearly look like roles/titles rather than names
    ROLE_TOKENS = [
        "developer",
        "engineer",
        "manager",
        "executive",
        "specialist",
        "consultant",
        "analyst",
        "architect",
        "scientist",
        "administrator",
        "sales",
        "marketing",
        "lead",
        "head",
        "officer",
        "coordinator",
        "business development",
    ]
    if any(tok in norm for tok in ROLE_TOKENS):
        return False
    # Reject company/org-like strings that often appear near the header in PDFs
    COMPANY_TOKENS = [
        "informatics",
        "technology",
        "technologies",
        "solutions",
        "systems",
        "services",
        "labs",
        "pvt",
        "ltd",
        "limited",
        "inc",
        "llp",
        "private",
    ]
    if any(tok in norm for tok in COMPANY_TOKENS):
        return False
    # A line that clearly looks like a generic section header should not be a name,
    # but allow ALL-CAPS names which often get misdetected as "headers".
    if _looks_like_header(c) and not c.isupper():
        return False
    if any(ch.isdigit() for ch in c):
        return False
    # letters + spaces + dot/apostrophe allowed
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{1,58}[A-Za-z]?", c):
        return False
    parts = [p for p in re.split(r"\s+", c) if p]
    # Require at least 2 tokens to avoid selecting skills like "Wordpress" as a "name".
    if not (2 <= len(parts) <= 4):
        return False
    if any(p.lower() in {"and", "or", "of", "the"} for p in parts):
        return False
    # Previously all‑caps was rejected; instead, rely on HEADER_BAD_NAMES/ROLE_TOKENS
    # so we can still accept names like "JOHN DOE" in the header.
    return True


def extract_name_from_header(resume_text: str) -> str:
    """
    Robust name detection from the top of the resume.
    Avoids headings and strips phone/email/urls.
    """
    if not resume_text:
        return ""
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    top = lines[:25]
    # Prefer the first plausible person-name line, after stripping contact noise and separators
    for ln in top:
        cand = _strip_contact_noise(ln)
        # keep only before common separators if present
        for sep in ("|", "•"):
            if sep in cand:
                cand = cand.split(sep, 1)[0].strip()
        cand = cand.strip()
        if _is_plausible_person_name(cand):
            return cand

    # Fallback: many PDFs split the name across consecutive single-token lines,
    # e.g. "GOURAV" / "DHIMAN". Combine the first plausible adjacent pair.
    def _is_name_token(tok: str) -> bool:
        raw = _strip_contact_noise(tok or "").strip()
        if not raw or " " in raw:
            return False
        # Keep only alphabetic core for classification (PDFs often add stray glyphs)
        core = re.sub(r"[^A-Za-z]+", "", raw)
        if len(core) < 2 or len(core) > 24:
            return False
        n = _norm_header(raw)
        if not n or n in _HEADER_BAD_NAMES or n in _COMMON_SECTION_HEADERS or n in _SKILLS_SECTION_HEADERS:
            return False
        if n in {"hindi", "english"}:
            return False
        if n in {"phone", "email", "location", "experience"}:
            return False
        if any(w in n for w in ("developer", "engineer", "manager", "skills", "languages")):
            return False
        return True

    for i in range(min(15, len(top) - 1)):
        t1 = top[i]
        t2 = top[i + 1]
        if _is_name_token(t1) and _is_name_token(t2):
            combined = f"{t1.strip()} {t2.strip()}"
            if _is_plausible_person_name(combined):
                return combined
    return ""


# ---------------------------------------------------------------------------
# Production name extraction: candidate-based scoring (no hardcoded bad-name list)
# ---------------------------------------------------------------------------

_nlp_ner = None


def _get_nlp_ner():
    """Lazy-load spaCy NER model. Optional: if not installed, NER is skipped."""
    global _nlp_ner
    if _nlp_ner is None:
        try:
            import spacy
            _nlp_ner = spacy.load("en_core_web_sm")
            logger.info("spaCy NER model loaded for name extraction.")
        except Exception as e:
            logger.warning("spaCy NER not available (run: python -m spacy download en_core_web_sm): %s", e)
            _nlp_ner = False  # mark as attempted
    return _nlp_ner if _nlp_ner else None


def _extract_person_spans(text: str) -> list[str]:
    """Return list of PERSON entities from NER. Returns [] if NER not available."""
    nlp = _get_nlp_ner()
    if not nlp:
        return []
    try:
        doc = nlp(text[:10000])
        return [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 4]
    except Exception as e:
        logger.debug("NER extract persons failed: %s", e)
        return []


# Generic institution/role/degree tokens used to penalise non-person candidates
_INSTITUTION_WORDS = {
    "department", "university", "college", "institute", "school", "faculty",
    "office of", "ministry", "bureau", "division", "corporation", "inc", "ltd",
}
_ROLE_WORDS = {
    "developer", "engineer", "manager", "executive", "specialist", "consultant",
    "analyst", "architect", "scientist", "administrator", "sales", "marketing",
    "lead", "head", "officer", "coordinator", "business development", "team",
    "management",  # e.g. "Team Management", "Incident Management"
    "designer", "web designer", "ui designer", "ux designer", "ui/ux designer",
}
# Education/degree phrases — never a person name
_EDUCATION_DEGREE_WORDS = {
    "bachelor", "bachelors", "master", "masters", "administration", "degree",
    "diploma", "phd", "mba", "btech", "mtech", "bsc", "msc", "bba", "mba",
    "business administration", "computer application", "arts", "science",
    "commerce", "engineering", "technology",
}


def extract_name_candidates(resume_text: str, email: str) -> list[str]:
    """
    Gather name candidates from NER, header lines, and email local-part.
    Structure-independent: does not rely on a fixed list of "bad" headers.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = (s or "").strip()
        if not s or len(s) > 60:
            return
        key = s.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(s)

    # 1) NER PERSON entities
    for person in _extract_person_spans(resume_text):
        add(person)

    # 2) Header lines (top 25) and lines around first email/phone
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    top = lines[:25]

    # 2a) Combine the very first two lines as "First Last" when they look like
    # separate name parts (e.g. "Santosh" / "Kumar"), so multi-line names
    # become a strong candidate.
    if len(top) >= 2:
        l1, l2 = top[0], top[1]
        # Single-token, alphabetic, capitalised words that are not headers/roles.
        def _is_name_token(tok: str) -> bool:
            tok = tok.strip()
            if " " in tok:
                return False
            if not tok or not tok[0].isalpha():
                return False
            if not tok[0].isupper():
                return False
            if not re.fullmatch(r"[A-Za-z][A-Za-z.'\-]{0,30}", tok):
                return False
            norm_tok = _norm_header(tok)
            if norm_tok in _HEADER_BAD_NAMES or norm_tok in _COMMON_SECTION_HEADERS:
                return False
            if any(w in norm_tok for w in _ROLE_WORDS):
                return False
            return True

        if _is_name_token(l1) and _is_name_token(l2):
            combined = f"{l1.strip()} {l2.strip()}"
            add(combined)
    email_re = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    contact_line_indices = set()
    for i, ln in enumerate(lines[:40]):
        if email_re.search(ln) or re.search(r"\+?\d[\d\s\-]{8,}", ln):
            contact_line_indices.add(i)
            for j in range(max(0, i - 1), min(len(lines), i + 2)):
                contact_line_indices.add(j)
    header_and_contact = sorted(set(list(range(len(top))) + [i for i in contact_line_indices if i < 50]))
    for i in header_and_contact:
        if i >= len(lines):
            break
        ln = lines[i]
        cleaned = _strip_contact_noise(ln)
        # Do not treat section headers as name candidates
        if _looks_like_header(cleaned) or _norm_header(cleaned) in _COMMON_SECTION_HEADERS or _norm_header(cleaned) in _SKILLS_SECTION_HEADERS:
            continue
        for sep in ("|", "•", "/", "–", "-"):
            if sep in cleaned:
                for part in cleaned.split(sep):
                    part_clean = part.strip()
                    n = _norm_header(part_clean)
                    if n in _HEADER_BAD_NAMES or n in _COMMON_SECTION_HEADERS or n in _SKILLS_SECTION_HEADERS:
                        continue
                    add(part_clean)
        add(cleaned)

    # 3) Email local-part derived
    if email:
        local = email.split("@", 1)[0]
        local = re.sub(r"[._\-]+", " ", local).strip()
        if local and any(ch.isalpha() for ch in local):
            parts = [p for p in local.split() if p]
            if 1 <= len(parts) <= 4:
                add(" ".join(w.capitalize() for w in parts))

    return candidates


def select_best_name(candidates: list[str], resume_text: str, email: str) -> tuple[str, bool]:
    """
    Score each candidate with generic rules; return (best_name, low_confidence).
    No hardcoded "bad names" list — uses institution/role tokens and shape/location.
    """
    if not candidates:
        # Fallback to email-derived only
        if email:
            local = email.split("@", 1)[0]
            local = re.sub(r"[._\-]+", " ", local).strip()
            if local and any(ch.isalpha() for ch in local):
                parts = [p for p in local.split() if p]
                if 1 <= len(parts) <= 4:
                    return (" ".join(w.capitalize() for w in parts), True)
        return ("", True)

    # If we have any plausible person-name candidates, only score those.
    # This prevents single-skill tokens like "Wordpress" from winning.
    plausible = [c for c in candidates if _is_plausible_person_name(c)]
    if plausible:
        candidates = plausible

    text_lower = (resume_text or "").lower()
    email_local = (email or "").split("@", 1)[0].lower()
    lines = [ln.strip() for ln in (resume_text or "").splitlines() if ln.strip()]
    top_10 = " ".join(lines[:10]).lower()

    best_name = ""
    best_score = -1.0

    for c in candidates:
        score = 0.0
        norm = _norm_header(c)
        if not norm:
            continue

        # Hard reject section headers / titles as names
        if norm in _HEADER_BAD_NAMES or norm in _COMMON_SECTION_HEADERS or norm in _SKILLS_SECTION_HEADERS:
            continue
        # Also reject generic section/skills headers
        if norm in _COMMON_SECTION_HEADERS or norm in _SKILLS_SECTION_HEADERS:
            continue
        # And reject anything that looks like a section header at the top of the resume
        # BUT allow plausible person names even if all-caps (common PDF formatting).
        if _looks_like_header(c) and not _is_plausible_person_name(c):
            continue

        # Strongly penalise pure technology phrases (e.g. "Entity Framework", "React .NET")
        if _is_core_tech_label(c):
            score -= 1.0

        # Penalise institution-like
        if any(w in norm for w in _INSTITUTION_WORDS):
            score -= 0.7
        # Penalise role-like (e.g. "Team Management", "Incident Management")
        if any(w in norm for w in _ROLE_WORDS):
            score -= 0.5
        # Penalise education/degree phrases (e.g. "Bachelors of Business Administration")
        if any(w in norm for w in _EDUCATION_DEGREE_WORDS):
            score -= 0.85
        # "X of Y" often means "Department of X" or "Bachelor of X", not a person
        if " of " in norm:
            score -= 0.6
        # Penalise all-caps long tokens (acronyms/headings)
        if c.isupper() and len(c) > 3:
            score -= 0.4
        # Penalise digits (except rare suffixes)
        if re.search(r"\d", c) and not re.search(r"\b(II|III|IV|Jr|Sr)\b", c, re.I):
            score -= 0.5
        # Penalise comma-separated city/state patterns (often locations)
        if "," in c and len(c.split()) <= 4:
            score -= 0.7
        # Penalise obvious location words
        if any(w in norm for w in ("punjab", "mohali", "chandigarh", "india", "delhi", "mumbai", "bangalore", "hyderabad")):
            score -= 0.6
        # Shape: 1–4 words, alphabetic, favour title‑case or capitalised tokens
        parts = [p for p in re.split(r"\s+", c) if p]
        if 1 <= len(parts) <= 4 and all(p[0].isupper() for p in parts if p):
            score += 0.4
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{1,58}[A-Za-z]?", c):
            score -= 0.2
        # Appears in top of doc
        if norm in top_10 or any(norm in _norm_header(ln) for ln in lines[:15]):
            score += 0.3
        # Overlap with email local-part (strong signal)
        for word in norm.split():
            if len(word) > 2 and word.lower() in email_local:
                score += 0.35
                break
        # NER gave this (we can't distinguish here; NER candidates are just in the list)
        # Prefer shorter, title-case
        if c.istitle() or (len(parts) == 2 and c[0].isupper()):
            score += 0.1

        if score > best_score:
            best_score = score
            best_name = c

    # Reject best candidate if score is too low OR is a known bad header
    if best_score < 0.2 or (best_name and _norm_header(best_name) in _HEADER_BAD_NAMES.union(_COMMON_SECTION_HEADERS).union(_SKILLS_SECTION_HEADERS)):
        best_name = ""

    low_confidence = best_score < 0.3 or not best_name
    if low_confidence and email:
        local = email.split("@", 1)[0]
        local = re.sub(r"[._\-]+", " ", local).strip()
        if local and any(ch.isalpha() for ch in local):
            parts = [p for p in local.split() if p]
            if 1 <= len(parts) <= 4:
                best_name = " ".join(w.capitalize() for w in parts)
                low_confidence = True
    return (best_name or "", low_confidence)


def _find_section_block(
    resume_text: str,
    headers: set[str],
    *,
    max_lines: int = 30,
) -> str:
    """Return a block of lines after the first matching header, or empty string."""
    lines = [ln.strip() for ln in (resume_text or "").splitlines()]
    if not lines:
        return ""
    start = None
    for i, ln in enumerate(lines):
        if _norm_header(ln) in headers:
            start = i + 1
            break
    if start is None:
        return ""
    out: list[str] = []
    blank_streak = 0
    for ln in lines[start:]:
        if not ln:
            blank_streak += 1
            if out and blank_streak >= 2:
                break
            continue
        blank_streak = 0
        if _looks_like_header(ln) and out:
            break
        out.append(ln)
        if len(out) >= max_lines:
            break
    return "\n".join(out).strip()


def _is_skills_header_line(line: str) -> bool:
    """
    Heuristic to detect skills-like section headers with light fuzziness.
    Covers variations like 'Technical Skills', 'Tech Stack', 'Core Technical Competencies', etc.
    """
    norm = _norm_header(line)
    if not norm:
        return False
    if norm in _SKILLS_SECTION_HEADERS:
        return True
    # Fuzzy: any header that clearly mentions skills / competencies / tech
    SKILL_KEYWORDS = (
        "skill",
        "skills",
        "competenc",
        "tech stack",
        "technology",
        "technologies",
        "tools",
        "stack",
    )
    if any(kw in norm for kw in SKILL_KEYWORDS):
        return True
    return False


def build_extraction_context(resume_text: str) -> tuple[str, str, str]:
    """
    Build structured context blocks for the LLM:
    - header_block: top contact/header area
    - skills_block: skills/tools/tech stack area if present
    - body_block: a larger snippet so skills/experience later in the doc still show up
    """
    lines = [ln.rstrip() for ln in (resume_text or "").splitlines()]
    header_block = "\n".join([ln for ln in lines[:35] if ln.strip()])[:1200]
    skills_block = _find_section_block(resume_text, _SKILLS_SECTION_HEADERS, max_lines=35)
    # Give the model more than 1500 chars; NuExtract is still limited by tokenizer max_length
    body_block = (resume_text or "")[:6000]
    return header_block.strip(), skills_block.strip(), body_block.strip()


_SKILL_CANONICAL = {
    "dotnet": ".NET",
    ".net": ".NET",
    "aspnet": "ASP.NET",
    "asp.net": "ASP.NET",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "reactjs": "React",
    "react js": "React",
    "react": "React",
    "nextjs": "Next.js",
    "next js": "Next.js",
    "next": "Next.js",
    "postgres": "PostgreSQL",
    "postgre": "PostgreSQL",
    "py torch": "PyTorch",
    "pytorch": "PyTorch",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "git hub": "GitHub",
    "github": "GitHub",
    "git lab": "GitLab",
    "gitlab": "GitLab",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "py": "Python",
    "c sharp": "C#",
    "c#": "C#",
    "sql server": "SQL Server",
    "ms sql server": "SQL Server",
}


def canonicalise_skill(skill: str) -> str:
    s = re.sub(r"\s+", " ", (skill or "").strip())
    if not s:
        return ""
    # Exclude methodology labels from technical skills
    if _norm_header(s) in {"agile", "scrum", "kanban"}:
        return ""
    key = re.sub(r"[^a-z0-9.+#]+", " ", s.lower()).strip()
    if key in _SKILL_CANONICAL:
        return _SKILL_CANONICAL[key]
    # preserve common acronyms
    if s.isupper() and len(s) <= 15:
        return s
    return _normalise_skill_token(s)


def _extra_tech_vocab_file_path() -> str:
    """Optional override via EXTRA_TECH_VOCAB_PATH; else backend/extra_tech_vocab.txt next to this module."""
    override = (os.getenv("EXTRA_TECH_VOCAB_PATH") or "").strip().strip('"')
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra_tech_vocab.txt")


def _read_extra_tech_vocab_terms() -> list[str]:
    """One skill or phrase per line from extra_tech_vocab.txt (lines starting with # ignored)."""
    path = _extra_tech_vocab_file_path()
    if not os.path.isfile(path):
        return []
    out: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.append(line)
    except OSError as exc:
        logger.warning("Could not read extra tech vocabulary %s: %s", path, exc)
    return out


def _build_tech_skill_vocab(skill_keywords: list[str]) -> set[str]:
    """
    Normalised vocabulary for technical-skill gating (languages, stacks, certs, etc.).
    Soft skills are rejected by a positive technical gate, not by an ever-growing blocklist.
    Merges terms from extra_tech_vocab.txt (see _extra_tech_vocab_file_path).
    """
    v: set[str] = set()
    for x in skill_keywords or []:
        t = str(x).strip()
        if not t:
            continue
        v.add(t.lower())
        v.add(_norm_header(t))
    for kw in _CORE_TECH_PRIMARY:
        if kw:
            v.add(kw.lower())
            v.add(_norm_header(kw))
    for cur, disp in _SKILL_CANONICAL.items():
        if cur:
            v.add(cur.lower())
            v.add(_norm_header(cur))
        if disp:
            v.add(disp.lower())
            v.add(_norm_header(disp))
    for line in _read_extra_tech_vocab_terms():
        t = str(line).strip()
        if not t:
            continue
        v.add(t.lower())
        v.add(_norm_header(t))
    return {x for x in v if x}


def _skill_vocab_hit(norm_skill: str, vocab: set[str]) -> bool:
    """
    norm_skill should be _norm_header(...) (spaces, lowercase, alnum only).
    Matches whole skill, multi-word phrases as substrings, or single-word terms with boundaries.
    """
    if not norm_skill:
        return False
    if norm_skill in vocab:
        return True
    for term in vocab:
        if not term or len(term) < 2:
            continue
        tn = _norm_header(term)
        if not tn:
            continue
        if tn == norm_skill:
            return True
        if " " in tn:
            if tn in norm_skill:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(tn)}(?![a-z0-9])", norm_skill):
            return True
    return False


def _skill_vocab_sliding_hit(norm_skill: str, vocab: set[str]) -> bool:
    """Match 2–4 word windows so phrases like 'Azure Machine Learning' hit known n-grams."""
    parts = [p for p in (norm_skill or "").split() if p]
    if len(parts) < 2:
        return False
    for n in (2, 3, 4):
        if n > len(parts):
            continue
        for i in range(len(parts) - n + 1):
            frag = " ".join(parts[i : i + n])
            if _skill_vocab_hit(frag, vocab):
                return True
    return False


def _passes_technical_skill_output(raw: str, vocab: set[str]) -> bool:
    """
    True if the label should appear in technical skills output.
    Uses vocabulary + tech-shaped tokens + certifications — not a soft-skill denylist.
    """
    s = _strip_skill_decorators(str(raw))
    if not s:
        return False
    if _is_education_like_phrase(s):
        return False
    if _is_percentage_metric(s):
        return False
    if _skill_is_noise(s):
        return False
    norm_low = _norm_header(s)
    canon = canonicalise_skill(s)
    cn = _norm_header(canon) if canon else ""
    if _skill_vocab_hit(norm_low, vocab) or (cn and _skill_vocab_hit(cn, vocab)):
        return True
    if _skill_vocab_sliding_hit(norm_low, vocab) or (cn and _skill_vocab_sliding_hit(cn, vocab)):
        return True
    if _is_core_tech_label(s):
        return True
    if any(ch in s for ch in ".+#/\\"):
        return True
    if re.search(r"(?<![a-z0-9])[a-z]{1,6}-\d{2,4}(?![a-z0-9])", norm_low):
        return True
    sl = s.strip().lower()
    if _looks_like_certification(sl) and re.search(
        r"\b(aws|azure|gcp|google cloud|kubernetes|microsoft|cisco|oracle|comptia|"
        r"hashicorp|salesforce|terraform|docker|cka|ckad|pmp|oscp)\b",
        sl,
    ):
        return True
    w = s.strip()
    parts = w.split()
    if len(parts) == 1 and len(w) <= 24:
        if re.match(r"^[A-Za-z][a-z0-9]*([A-Z][a-z0-9]*)+$", w):
            return True
        if w.isupper() and w.isalpha() and 2 <= len(w) <= 6:
            if w in {"HR", "CV", "CEO", "VP"}:
                return False
            return True
    if len(parts) > 4:
        return False
    return False


def extract_skills_from_text(resume_text: str) -> list[str]:
    """
    Extract skills as explicitly written in the resume (e.g., under 'SKILLS').
    Avoids calling the LLM; designed to be fast and deterministic.
    """
    text = (resume_text or "").strip()
    if not text:
        return []

    # Robust carve-out: grab substring after the first SKILLS *header line*
    # until the next major header. (Avoid matching "skills" inside sentences.)
    carved_block = ""
    header_idx = None
    header_end = None
    for m in re.finditer(r"(?im)^\s*skills\s*:?\s*$", text):
        header_idx = m.start()
        header_end = m.end()
        break
    if header_idx is None:
        # Sometimes headers are on the same line with other headers ("EDUCATION SKILLS")
        # or in all-caps footer blocks; handle explicit header lines in the first 120 lines.
        for m in re.finditer(r"(?im)^\s*[a-z &]+\bskills\b[a-z &]*\s*$", text):
            # Ensure it looks like a header line, not a sentence
            line = text[m.start() : m.end()].strip()
            if len(line) <= 40 and _looks_like_header(line):
                header_idx = m.start()
                header_end = m.end()
                break
    if header_end is not None:
        rest = text[header_end:]
        m1 = re.search(r"(?is)\b(summary|educ\w*|projects|experience|training|certifications)\b", rest)
        carved_block = (rest[: m1.start()] if m1 else rest)[:1800].strip()

    # Fallback: many PDFs put the actual skill list near the top (before headers),
    # and later repeat "EDUCATION EXPERIENCE SKILLS PROJECTS" in a footer.
    if not carved_block or len(carved_block) < 25:
        top_lines = [ln.strip() for ln in text.splitlines()[:45] if ln.strip()]
        strip: list[str] = []
        started = False
        for ln in top_lines:
            norm = _norm_header(ln)
            # Start collecting after a line that clearly indicates skills (header-like)
            if not started and ("skills" in norm) and _looks_like_header(ln):
                started = True
                continue
            if not started:
                continue
            # Stop at the next major section header
            if norm in {"experience", "projects", "education", "extra curicular", "extra curricular"}:
                break
            # Candidate skill line: short, low symbol ratio, not an email/phone/year
            if len(ln) > 30:
                continue
            if re.search(r"@\w+|\b\d{4}\b|\+?\d[\d\s\-]{8,}", ln):
                continue
            if not re.search(r"[a-zA-Z]", ln):
                continue
            if _symbol_ratio(ln) > 0.25:
                continue
            strip.append(ln)
        if len(strip) >= 5:
            carved_block = "\n".join(strip)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Find all skills-like section headers (not just the first one)
    section_blocks: list[list[str]] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if _is_skills_header_line(ln):
            start = i + 1
            section: list[str] = []
            blank_streak = 0
            for j in range(start, n):
                cur = lines[j]
                if not cur:
                    blank_streak += 1
                    if section and blank_streak >= 2:
                        break
                    continue
                blank_streak = 0
                if _looks_like_header(cur) and section:
                    break
                section.append(cur)
                if len(section) >= 80:
                    break
            if section:
                section_blocks.append(section)
            i = j
        else:
            i += 1

    # Fallback for PDFs where extraction collapses line breaks:
    # use regex to carve out a SKILLS-like block until the next major section.
    if not section_blocks:
        m = re.search(
            r"(?is)\b(skills|technical skills|tech stack|tools\s*&?\s*technologies)\b\s*:?\s*(.+?)\s*(\b(summary|educ\w*|projects|experience|training|certifications)\b|$)",
            text,
        )
        if m:
            block = m.group(2) or ""
            # Create a pseudo section block from the captured substring
            section_blocks = [[ln.strip() for ln in re.split(r"[\n\r]+", block) if ln.strip()]]
        elif carved_block:
            section_blocks = [[ln.strip() for ln in re.split(r"[\n\r]+", carved_block) if ln.strip()]]
        else:
            return []

    # Join all skill section lines and tokenize using the strict cleaner.
    section_joined = "\n".join(["\n".join(sec) for sec in section_blocks if sec]).strip()
    # Do not let a narrow "skills:" carve-out replace a full parsed section (common PDF bug).
    if section_joined:
        section_text = section_joined
    elif carved_block:
        section_text = carved_block
    else:
        section_text = ""
    if not section_text:
        return []

    # Candidate collection: prefer explicit skills section, but also harvest from
    # experience/projects when PDFs are messy or skills are embedded elsewhere.
    def _explode_compact_skill_line(tok: str) -> list[str]:
        """
        Split compact lines like 'HTML CSS JS' into individual tokens safely.
        Only applies when the token looks like a short list of short words.
        """
        s = _collapse_whitespace(tok or "")
        if not s or " " not in s:
            return [tok]
        parts = [p for p in s.split(" ") if p]
        if 2 <= len(parts) <= 5 and all(1 < len(p) <= 12 for p in parts):
            # Avoid splitting sentences: require low symbol ratio and no verbs.
            if _symbol_ratio(s) <= 0.18 and not re.search(
                r"\b(created|developed|implemented|managed|responsible)\b",
                _norm_header(s),
            ):
                return parts
        return [tok]

    def _collect_tokens(txt: str, *, cap: int) -> list[str]:
        toks: list[str] = []
        for t in _tokenize_skill_candidates(txt or ""):
            for x in _explode_compact_skill_line(t):
                x = _collapse_whitespace(x)
                if x:
                    toks.append(x)
            if len(toks) >= cap:
                break
        return toks[:cap]

    # Source A: skills section (highest precision)
    candidates = _collect_tokens(section_text, cap=260)

    # Source B: experience/projects — ONLY when no real skills section was found.
    # If the resume has an explicit Skills / Technical Skills block, harvesting from
    # experience/projects pulls narrative words ("For", "Registration", verbs, etc.).
    exp_txt = ""
    proj_txt = ""
    if not section_blocks:
        blocks = _get_context_blocks_for_evidence(text)
        exp_txt = _find_section_block(
            text,
            {"experience", "work experience", "work history", "professional experience"},
            max_lines=140,
        ) or ""
        proj_txt = _find_section_block(text, {"projects"}, max_lines=140) or ""
        if len(candidates) < 8:
            candidates += _collect_tokens(exp_txt, cap=180)
            candidates += _collect_tokens(proj_txt, cap=140)

    # Normalize + pre-filter (lightweight). Final acceptance is evidence-based.
    normalized: list[str] = []
    seen_norm: set[str] = set()
    for tok in candidates:
        norm = _normalize_skill_token(tok)
        if not norm:
            continue
        if _skill_is_noise(norm):
            continue
        if _norm_header(norm) in _COMMON_SECTION_HEADERS:
            continue
        if _looks_like_header(norm):
            continue
        if not _skill_token_is_valid(norm):
            continue
        key = norm.lower()
        if key in seen_norm:
            continue
        seen_norm.add(key)
        normalized.append(norm)

    # Evidence-based acceptance & weighting (generalizes across PDF layouts).
    # Use the skills-section-only list as a precision anchor.
    section_anchor = _collect_tokens(section_text, cap=260)
    section_anchor_norm = []
    for t in section_anchor:
        n = _normalize_skill_token(t)
        if n and not _skill_is_noise(n) and _skill_token_is_valid(n):
            section_anchor_norm.append(n)

    # Low-confidence detection based on overall text quality.
    c, _ = _compute_extraction_confidence(text)
    filtered, _weights = _compute_skill_weights(
        normalized,
        resume_text=text,
        section_skills=section_anchor_norm,
        extraction_low_confidence=bool(c < 0.5),
    )
    return filtered


def _get_context_blocks_for_evidence(resume_text: str) -> dict:
    """
    Extract relevant blocks for evidence checks. Internal use only.
    Returns lowercase strings for matching.
    """
    skills_block = _find_section_block(resume_text, _SKILLS_SECTION_HEADERS, max_lines=60)
    exp_block = _find_section_block(
        resume_text,
        {
            "experience",
            "work experience",
            "work experiences",
            "work history",
            "professional experience",
            "employment history",
        },
        max_lines=120,
    )
    proj_block = _find_section_block(resume_text, {"projects"}, max_lines=120)
    return {
        "skills": (skills_block or "").lower(),
        "experience": (exp_block or "").lower(),
        "projects": (proj_block or "").lower(),
        "all": (resume_text or "").lower(),
    }


def _skill_token_is_valid(tok: str) -> bool:
    """
    Strict filter for skill tokens (reject narrative fragments and junk).
    """
    if not tok:
        return False
    s = _strip_footnote_numbers(str(tok)).strip()
    if len(s) < 2 or len(s) > 48:
        return False
    # Reject sentence-like fragments (too many words)
    if len(s.split()) > 8:
        return False
    # Reject obvious narrative verbs / durations
    low = _norm_header(s)
    if re.search(r"\b(months?|yrs?|years?)\b", low):
        return False
    if re.search(
        r"\b(created|developed|performed|integrated|optimized|designed|implemented|managed|led|built|delivered|debugg|responsible)\w*\b",
        low,
    ):
        return False
    # Digits are usually noise; allow only cert-like codes (AZ-900) or common tech tokens (C++/.NET)
    if re.search(r"\d", s):
        if not re.search(r"\b[a-z]{1,4}-?\d{2,4}\b", s, flags=re.I):
            return False
    # Special chars: allow for C++, C#, .NET, Node.js, etc.
    if re.search(r"[^A-Za-z0-9 .+#\-]", s):
        return False
    return True


def _count_skill_occurrences(skill: str, text: str) -> int:
    """
    Count approximate occurrences of a skill in text using word boundaries for short tokens.
    """
    if not skill or not text:
        return 0
    key = re.sub(r"[^a-z0-9]+", " ", skill.lower()).strip()
    if not key:
        return 0
    # single short token -> boundary
    if len(key) <= 4 and " " not in key:
        return len(re.findall(rf"\b{re.escape(key)}\b", text))
    # multiword -> count phrase occurrences in normalized text
    parts = [p for p in key.split() if p]
    if not parts:
        return 0
    phrase = " ".join(parts)
    return len(re.findall(rf"\b{re.escape(phrase)}\b", text))


def _compute_skill_weights(
    skills: list[str],
    *,
    resume_text: str,
    section_skills: list[str],
    extraction_low_confidence: bool,
) -> tuple[list[str], dict]:
    """
    Evidence-based weighting and acceptance for skills.
    Returns (filtered_skills, weight_by_skill_lower).
    """
    blocks = _get_context_blocks_for_evidence(resume_text)
    weights: dict[str, int] = {}
    out: list[str] = []
    seen: set[str] = set()

    section_keys = {canonicalise_skill(s).lower() for s in (section_skills or []) if canonicalise_skill(s)}

    for raw in skills or []:
        s0 = canonicalise_skill(str(raw))
        if not s0:
            continue
        s0 = _strip_footnote_numbers(s0)
        if not _skill_token_is_valid(s0):
            continue

        k = s0.lower()
        if k in seen:
            continue

        in_skills_section = k in section_keys or _count_skill_occurrences(s0, blocks["skills"]) > 0
        in_exp_or_proj = (_count_skill_occurrences(s0, blocks["experience"]) > 0) or (
            _count_skill_occurrences(s0, blocks["projects"]) > 0
        )
        total_occ = _count_skill_occurrences(s0, blocks["all"])

        # Acceptance rules (generalized, but avoid proper-noun noise):
        # - accept if in skills section OR appears in exp/projects
        # - otherwise require repeated occurrence AND that it looks like a technical token
        #   (core tech list, certification-like, or has tech punctuation like . / + #)
        low = _norm_header(s0)
        core_allow = {
            "html",
            "css",
            "javascript",
            "typescript",
            "react",
            "angular",
            "vue",
            "next",
            "next js",
            "nextjs",
            "node",
            "node js",
            "nodejs",
            "wordpress",
            "shopify",
        }
        looks_techy = (
            _is_core_tech_label(s0)
            or (low in core_allow)
            or _looks_like_certification(low)
            or any(ch in s0 for ch in (".", "#", "+", "/"))
        )
        # Trust tokens that came from explicit skills-section parsing (section_keys), even if
        # they are not in the small "looks_techy" shortcut list (e.g. Figma, Prisma, Kafka).
        accept = (
            in_exp_or_proj
            or (k in section_keys)
            or (in_skills_section and looks_techy)
            or (total_occ >= 2 and looks_techy)
        )
        if not accept:
            continue

        w = 0
        if in_exp_or_proj:
            w += 2
        if in_skills_section:
            w += 1
        if total_occ <= 1 and not in_exp_or_proj and not in_skills_section:
            w -= 1

        # If low confidence, require stronger evidence and avoid tool-dump lists
        if extraction_low_confidence and w <= 0:
            continue

        weights[k] = w
        seen.add(k)
        out.append(s0)

    # For low-confidence resumes, cap the number of skills to reduce noise.
    if extraction_low_confidence and out:
        out = sorted(out, key=lambda s: (weights.get(s.lower(), 0), _count_skill_occurrences(s, blocks["all"])), reverse=True)[:35]

    return out, weights

def extract_location_from_text(resume_text: str) -> str:
    """
    Best-effort location extraction from the top of the resume.
    Looks for a city/state-like line and strips emails/phones/links.
    """
    if not resume_text:
        return ""
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    email_re = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    phone_re = r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){6,14}\d"

    for i, ln in enumerate(lines[:12]):
        low = ln.lower()
        if any(x in low for x in ("linkedin", "github", "http://", "https://", "www.")):
            continue

        cleaned = re.sub(email_re, "", ln)
        cleaned = re.sub(phone_re, "", cleaned)
        # Keep text before common separators
        for sep in ("•", "|"):
            if sep in cleaned:
                cleaned = cleaned.split(sep, 1)[0]
        cleaned = cleaned.strip(" -•|,")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue

        # Do not misclassify likely names as locations in the header area (common in PDFs)
        if i <= 2 and _is_plausible_person_name(cleaned):
            continue

        # Remove long pin codes if present (India etc.)
        cleaned = re.sub(r"\b\d{5,6}\b", "", cleaned).strip(" -•|,")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Heuristic: location often contains a comma or looks like "City State"
        if len(cleaned) < 3 or len(cleaned) > 60:
            continue
        if "@" in cleaned:
            continue
        if sum(ch.isdigit() for ch in cleaned) > 0:
            continue
        if not re.search(r"[a-zA-Z]", cleaned):
            continue

        # Avoid treating roles/titles as "location"
        norm = _norm_header(cleaned)
        if any(tok in norm for tok in ("developer", "engineer", "intern", "manager", "analyst", "designer", "consultant")):
            continue
        if _looks_like_header(cleaned):
            continue
        if "," in cleaned or len(cleaned.split()) in (2, 3):
            return cleaned

    return ""


def validate_and_repair_extraction(extracted: dict, resume_text: str) -> tuple[dict, list[str]]:
    """
    Validate key extracted fields against the (normalized) resume_text and repair
    common cross-field conflicts. Returns (extracted, warnings_added).
    Does NOT change API shape; only adjusts values and appends warning codes.
    """
    out = dict(extracted or {})
    warnings: list[str] = []

    text = resume_text or ""
    top_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:20]
    top_blob = " ".join(top_lines).lower()

    name = (out.get("name") or "").strip()
    location = (out.get("location") or "").strip()
    email = (out.get("email") or "").strip()

    # 1) Repair name/location swaps and demote implausible names.
    def _looks_like_location(s: str) -> bool:
        if not s:
            return False
        norm = _norm_header(s)
        if "," in s and 2 <= len(s.split()) <= 5:
            return True
        # small list of geo tokens (lightweight, not meant to be exhaustive)
        if any(w in norm for w in ("india", "punjab", "delhi", "mumbai", "bangalore", "hyderabad", "chandigarh")):
            return True
        return False

    def _looks_like_role(s: str) -> bool:
        if not s:
            return False
        norm = _norm_header(s)
        return any(
            tok in norm
            for tok in (
                "developer",
                "engineer",
                "intern",
                "manager",
                "analyst",
                "designer",
                "consultant",
                "software development",
                "development",
            )
        )

    if name and (
        (not _is_plausible_person_name(name))
        or (_norm_header(name) in {"hindi", "english", "language", "languages"})
        or _looks_like_location(name)
        or _looks_like_role(name)
    ):
        warnings.append("name_implausible")
        # Clear so fallbacks can populate a better value.
        out["name"] = ""
        # Try to recover from header region: pick first plausible person-name line.
        header_name = extract_name_from_header(text)
        if header_name and _is_plausible_person_name(header_name):
            out["name"] = header_name[:60]
            name = out["name"]
            warnings.append("name_repaired_from_header")
        if not (out.get("name") or "").strip():
            # Strong fallback for PDFs that split names into consecutive ALL-CAPS tokens.
            # Example:
            #   GOURAV
            #   DHIMAN
            stop = {"hindi", "english", "phone", "email", "location", "experience", "languages", "skills"}
            hdr_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()][:18]
            best = ""
            for i in range(len(hdr_lines) - 1):
                a = re.sub(r"[^A-Za-z]+", "", hdr_lines[i]).strip()
                b = re.sub(r"[^A-Za-z]+", "", hdr_lines[i + 1]).strip()
                if len(a) < 3 or len(b) < 3:
                    continue
                if not (a.isupper() and b.isupper()):
                    continue
                na = _norm_header(a)
                nb = _norm_header(b)
                if na in stop or nb in stop:
                    continue
                if na in _COMMON_SECTION_HEADERS or na in _SKILLS_SECTION_HEADERS:
                    continue
                if nb in _COMMON_SECTION_HEADERS or nb in _SKILLS_SECTION_HEADERS:
                    continue
                cand = f"{a} {b}"
                if _is_plausible_person_name(cand):
                    best = cand
                    break
            if best:
                out["name"] = best[:60]
                name = out["name"]
                warnings.append("name_repaired_from_caps_pair")

        if (not (out.get("name") or "").strip()) and email:
            # Fallback: derive from email local-part (better than persisting a header label)
            local = email.split("@", 1)[0]
            local = re.sub(r"[._\\-]+", " ", local).strip()
            if local and any(ch.isalpha() for ch in local):
                parts = [p for p in local.split() if p]
                if 1 <= len(parts) <= 4:
                    out["name"] = " ".join(w.capitalize() for w in parts)[:60]
                    name = out["name"]
                    warnings.append("name_repaired_from_email")

    # Swap if strong location/name conflict detected.
    if name and location:
        if _looks_like_location(name) and _is_plausible_person_name(location):
            out["name"], out["location"] = location, name
            warnings.append("name_swapped_with_location")
            name, location = out["name"], out["location"]

    # If location looks like a role, clear it (better blank than wrong).
    if location and _looks_like_role(location):
        out["location"] = ""
        warnings.append("location_cleared_role_like")

    # 2) Skills sanity: if we extracted *zero* skills but the text has a skills header,
    # mark it so API layer can choose safer behavior later (without changing response).
    skills = out.get("skills") or []
    if (not skills) and ("skills" in top_blob or re.search(r"(?im)^\\s*skills\\b", text)):
        warnings.append("skills_missing_despite_header")

    # De-dup warning merges are handled by caller; return the new warnings.
    return out, warnings


# ---------------------------------------------------------------------------
# Resume extraction (NuExtract)
# ---------------------------------------------------------------------------

def extract_resume(resume_text: str) -> dict:
    """Extract resume data — LLM first, regex fallback."""

    # Normalize first to reduce PDF variability for all downstream logic.
    normalized_text, norm_meta = normalize_resume_text(resume_text or "")

    # Internal confidence + sanitization controls (not exposed in API)
    extraction_confidence, _conf_meta = _compute_extraction_confidence(normalized_text or resume_text)
    extraction_low_confidence = extraction_confidence < 0.5

    # --- regex baseline ---
    name = "Unknown"
    email = ""
    phone = ""
    location = ""
    skills: list[str] = []
    education: list[str] = []
    projects: list[str] = []
    experience_summary = ""
    experience_years = 0.0
    work_experience: list[dict] = []
    summary = ""
    companies_worked_at: list[str] = []
    current_role = ""
    important_keywords: list[str] = []
    key_skills: list[str] = []

    extraction_warnings: list[str] = []
    if extraction_low_confidence:
        extraction_warnings.append("low_extraction_confidence")
    if norm_meta.get("actions"):
        extraction_warnings.append("text_normalized")

    # Sanitised text used for token validation and support checks
    # (keeps original resume_text untouched for any caller needs)
    base_text = normalized_text or (resume_text or "")
    resume_text_sanitised = _drop_symbol_heavy_lines(base_text)
    if extraction_low_confidence:
        # Aggressive cleanup for noisy PDFs
        resume_text_sanitised = _drop_symbol_heavy_lines(resume_text_sanitised, max_ratio=0.32)

    # Email and phone first (needed for name candidate scoring)
    email_match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", base_text)
    if email_match:
        email = email_match.group(1)

    phone_match = re.search(r"(\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})", base_text)
    if phone_match:
        phone = phone_match.group(1)
    else:
        phone_match = re.search(r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){6,14}\d", base_text)
        if phone_match:
            phone = re.sub(r"\s+", " ", phone_match.group(0)).strip()

    # Production name extraction: candidate-based scoring (no hardcoded bad-name list)
    try:
        name_candidates = extract_name_candidates(base_text, email)
        best_name, low_confidence = select_best_name(name_candidates, base_text, email)
        if best_name:
            name = best_name[:60]
        if low_confidence and best_name:
            extraction_warnings.append("name_low_confidence")
    except Exception as e:
        logger.warning("Candidate-based name extraction failed: %s", e)
        header_name = extract_name_from_header(base_text)
        if header_name:
            name = header_name[:60]

    # Many resumes have location in the top few lines
    location = extract_location_from_text(resume_text_sanitised or base_text)

    # If the chosen name looks like a location but the extracted location looks like a person name,
    # swap them (common when PDFs put NAME in caps and city/state in mixed case).
    if name and location:
        name_norm = _norm_header(name)
        if ("," in name or any(w in name_norm for w in ("punjab", "mohali", "chandigarh", "india"))) and _is_plausible_person_name(location):
            extraction_warnings.append("name_swapped_with_location")
            name, location = location, name

    # Comprehensive technical skills list (~500+ items)
    skill_keywords = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "go", "rust", "kotlin",
        "c", "c++", "c#", "php", "ruby", "swift", "r", "scala", "groovy",
        "perl", "lua", "elixir", "haskell", "f#", "clojure", "erlang",
        "objective-c", "julia", "sql",

        # Web Frameworks
        "react", "angular", "vue", "nextjs", "next.js", "svelte", "ember", "backbone",
        "django", "flask", "fastapi", "spring", "express", "nestjs", "laravel",
        "rails", "asp.net", "asp net", ".net", "dotnet",
        "graphql", "apollo", "redux", "nuxt", "nuxt.js", "gatsby", "prisma", "strapi",

        # Databases
        "postgresql", "mysql", "mongodb", "dynamodb", "cassandra", "redis",
        "elasticsearch", "firestore", "oracle", "sqlite", "mariadb", "couchdb",
        "neo4j", "memcached", "solr", "snowflake", "bigquery",

        # Cloud Platforms
        "aws", "azure", "gcp", "google cloud", "digitalocean", "heroku",
        "ibm cloud", "oracle cloud", "linode", "vultr", "aws lambda",
        "azure functions", "google cloud functions", "cloudflare", "vercel",
        "netlify", "render",

        # DevOps & Tools
        "docker", "kubernetes", "jenkins", "gitlab", "github", "git", "terraform",
        "nginx", "ansible", "vagrant", "prometheus", "grafana", "elk", "splunk",
        "datadog", "newrelic", "circleci", "travis", "github actions",

        # Enterprise Platforms
        "salesforce", "servicenow", "sap", "workday", "netsuite",
        "dynamics 365", "jira", "confluence", "sharepoint", "tableau",
        "figma", "sketch", "miro", "zeplin",
        "powerbi", "looker", "qlik",

        # Mobile Development
        "react native", "flutter", "xamarin", "cordova",
        "phonegap", "ionic", "swiftui", "jetpack compose",

        # Testing Frameworks
        "jest", "pytest", "selenium", "junit", "testng", "mockito", "mocha",
        "jasmine", "cypress", "puppeteer", "webdriver", "rspec", "cucumber",
        "postman", "jmeter",

        # Big Data & Analytics
        "spark", "hadoop", "hive", "pig", "airflow", "dbt",
        "kafka", "beam", "flink", "presto", "drill", "impala", "sqoop",

        # ML & Data Science
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "scipy", "matplotlib", "seaborn", "jupyter", "anaconda", "mlflow",
        "huggingface", "xgboost", "lightgbm",
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "data science", "feature engineering",

        # Message Queues
        "rabbitmq", "activemq", "nats", "zeromq",
        "nservicebus", "masstransit",

        # Front-end Tools
        "webpack", "vite", "parcel", "rollup", "gulp", "grunt", "npm", "yarn",
        "pnpm", "bower", "npm scripts", "lerna",

        # Version Control & CI
        "git", "svn", "mercurial", "perforce", "github", "gitlab", "bitbucket",
        "gitea", "gitbucket",

        # AWS Certifications
        "aws certified", "aws solutions architect", "aws developer",
        "aws sysops", "aws devops", "aws data analytics", "aws cloud practitioner",
        "aws associate",

        # Azure Certifications
        "azure certified", "azure administrator", "azure developer",
        "azure solutions architect", "azure data engineer", "azure ai",
        "az-900", "az-104",

        # Google Cloud Certifications
        "google cloud certified", "gcp associate", "gcp professional",
        "cloud architect", "cloud engineer",

        # Kubernetes Certifications
        "cka", "ckad", "kubernetes certified",

        # Other Certifications
        "pmp", "scrum", "cissp", "oscp", "rhce", "comptia", "security+",
        "lpic", "giac", "cism",

        # IDE & Editors
        "intellij", "vs code", "visual studio", "eclipse", "pycharm", "webstorm",
        "sublime text", "atom", "vim", "emacs",

        # OS & Environments
        "linux", "ubuntu", "centos", "windows", "macos", "solaris",
        "bsd", "rhel",

        # Other Common Skills
        "microservices", "rest api", "soap", "grpc", "websockets",
        "mqtt", "socket.io", "html", "css", "tailwind", "bootstrap",
        "node", "nodejs", "excel", "powerbi",
    ]
    _tech_vocab = _build_tech_skill_vocab(skill_keywords)
    # --- Skills extraction (cleaned, context-aware) ---
    base_text = resume_text_sanitised or resume_text
    resume_lower = (base_text or "").lower()

    # Parse explicit SKILLS section (highest precision source)
    try:
        section_skills = extract_skills_from_text(base_text)
    except Exception as exc:
        logger.debug("Skills section parse failed: %s", exc)
        section_skills = []

    # Evidence text blocks (restrict skill token mining to high-signal areas)
    blocks = _get_context_blocks_for_evidence(base_text)
    skills_text_parts = [blocks.get("skills", "")]
    # Extract explicit Tech lines from projects to capture stacks without pulling narrative
    proj_raw = blocks.get("projects", "")
    if proj_raw:
        for ln in proj_raw.splitlines():
            lns = ln.strip()
            if not lns:
                continue
            if lns.lower().startswith("tech:"):
                skills_text_parts.append(lns.split(":", 1)[1].strip())
    # Lightweight experience stack hints: keep only lines that look like comma-separated stacks
    exp_raw = blocks.get("experience", "")
    if exp_raw:
        for ln in exp_raw.splitlines():
            if "," in ln and len(ln) <= 160:
                skills_text_parts.append(ln.strip())
    skills_text = "\n".join([p for p in skills_text_parts if p]).strip()

    # Candidate tokens from cleaned tokenization + known vocabulary scan
    candidates = _tokenize_skill_candidates(skills_text)

    # Known technical vocabulary baseline (from existing list)
    vocab = {str(x).strip().lower() for x in (skill_keywords or []) if str(x).strip()}
    vocab.update({str(x).strip().lower() for x in _CORE_TECH_PRIMARY})
    vocab.update({"asp.net core", "azure devops", "sql server", "entity framework", "rest api", "microservices", "ci/cd"})

    detected: list[str] = []
    for tok in candidates:
        tok = _strip_footnote_numbers(tok)
        if not tok:
            continue
        if len(tok) < 2 or len(tok) > 52:
            continue
        if re.fullmatch(r"[0-9]+", tok):
            continue
        norm = _normalize_skill_token(tok)
        if not norm or _skill_is_noise(norm):
            continue

        norm_low = _norm_header(norm)

        # Validation: accept if in known vocab OR repeated OR appears in SKILLS/EXPERIENCE blocks
        in_vocab = norm_low in vocab
        occ_all = _count_skill_occurrences(norm, blocks.get("all", ""))
        in_skills = _count_skill_occurrences(norm, blocks.get("skills", "")) > 0
        in_exp = _count_skill_occurrences(norm, blocks.get("experience", "")) > 0

        if not (in_vocab or occ_all >= 2 or in_skills or in_exp):
            continue

        detected.append(norm)

    # Merge section skills + detected candidates, then evidence-weight + filter
    merged_raw = list(dict.fromkeys([_normalize_skill_token(s) for s in (section_skills or []) if _normalize_skill_token(s)] + detected))
    skills, skill_weights = _compute_skill_weights(
        merged_raw,
        resume_text=base_text,
        section_skills=section_skills or [],
        extraction_low_confidence=extraction_low_confidence,
    )

    # Project-specific noise: drop tokens that exist only in projects block
    prj = blocks.get("projects", "")
    expb = blocks.get("experience", "")
    skb = blocks.get("skills", "")
    _section_skill_canon_early = {
        canonicalise_skill(s).lower() for s in (section_skills or []) if canonicalise_skill(s)
    }
    filtered2: list[str] = []
    for s in skills or []:
        in_prj = _count_skill_occurrences(s, prj) > 0
        in_exp = _count_skill_occurrences(s, expb) > 0
        in_skl = _count_skill_occurrences(s, skb) > 0
        if in_prj and not in_exp and not in_skl:
            sk0 = canonicalise_skill(s)
            if sk0 and sk0.lower() in _section_skill_canon_early:
                filtered2.append(s)
                continue
            # exclude from core list (low relevance)
            continue
        filtered2.append(s)
    skills = filtered2

    # Education extraction: prefer EDUCATION section to avoid false positives
    # like "engineering" in role descriptions or "certificate" in cert lists.
    degree_keywords = [
        "bachelor",
        "master",
        "phd",
        "b.tech",
        "m.tech",
        "diploma",
        "b.sc",
        "m.sc",
        "engineering",
        "certificate",
    ]
    edu_block = _find_section_block(resume_text, {"education"}, max_lines=40)
    edu_text = (edu_block or resume_text or "").lower()
    for degree in degree_keywords:
        if degree.lower() in edu_text:
            education.append(degree.title())
    education = list(dict.fromkeys(education))

    if "project" in resume_lower:
        text_lines = resume_text.splitlines()
        collect = False
        for line in text_lines:
            if "project" in line.lower():
                collect = True
                continue
            if collect:
                if not line.strip():
                    break
                projects.append(line.strip())

    # --- LLM extraction (optional; can be slow on CPU) ---
    # For much faster uploads, set USE_LLM_EXTRACTION=0 (default).
    use_llm = (os.getenv("USE_LLM_EXTRACTION", "0") or "").strip().lower() in {"1", "true", "yes"}
    if use_llm:
        try:
            header_block, skills_block, body_block = build_extraction_context(resume_text)
            prompt = (
            "You are extracting factual resume data from text.\n\n"
            "RULES:\n"
            "- Use ONLY the provided text. If missing, return \"\" or [].\n"
            "- Candidate name MUST be a real person name (2-15 words). Do NOT return headings like RESUME/CV or job titles.\n"
            "- Do NOT include phone/email/links in the name.\n"
            "\n"
            "TECHNICAL SKILLS DEFINITION:\n"
            "Technical skills are: programming languages, frameworks, databases, cloud platforms, tools, libraries, and certifications.\n"
            "\n"
            "DO EXTRACT as skills:\n"
            "✓ 'Python', 'JavaScript', 'React', 'AWS', 'Docker', 'PostgreSQL'\n"
            "✓ 'AWS Certified Solutions Architect', 'Kubernetes Administrator'\n"
            "✓ 'Machine Learning', 'Microservices', 'REST API'\n"
            "\n"
            "DO NOT EXTRACT as skills (these go elsewhere):\n"
            "✗ Soft skills: 'Communication', 'Teamwork', 'Leadership', 'Problem-solving'\n"
            "✗ Methodologies: 'Agile', 'Scrum', 'Kanban'\n"
            "✗ Non-technical: 'Sales', 'Marketing', 'Business Development'\n"
            "✗ Educational degrees: 'Bachelor of Science' (these go in 'education' field)\n"
            "✗ Job titles as skills: 'Software Engineer', 'Data Analyst'\n"
            "✗ Generic terms: 'Software Development', 'Programming', 'Technology'\n"
            "\n"
            "KEY_SKILLS vs SKILLS distinction:\n"
            "- 'key_skills': Your TOP 3-5 core/primary technologies (what the candidate specializes in)\n"
            "- 'skills': All technical skills AND certifications mentioned (5-20 items total)\n"
            "Priority for key_skills: pick the most frequently mentioned, most recent, or most relevant to current role\n"
            "\n"
            "Return EXACTLY one JSON object. No markdown, no extra text.\n\n"
            "Return this JSON schema:\n"
            "{\n"
            '  "name": "",\n'
            '  "email": "",\n'
            '  "phone": "",\n'
            '  "location": "",\n'
            '  "skills": [],\n'
            '  "experience_years": 0,\n'
            '  "experience_summary": "",\n'
            '  "work_experience": [\n'
            '    {"company": "", "role": "", "start_date": "", "end_date": ""}\n'
            "  ],\n"
            '  "education": [],\n'
            '  "summary": "",\n'
            '  "companies_worked_at": [],\n'
            '  "current_role": "",\n'
            '  "important_keywords": [],\n'
            '  "key_skills": []\n'
            "}\n\n"
            f"HEADER BLOCK:\n{header_block}\n\n"
            f"SKILLS/COMPETENCIES BLOCK:\n{skills_block}\n\n"
            f"BODY SNIPPET:\n{body_block}\n"
            )

            tokenizer = _get_extract_tokenizer()
            model = _get_extract_model()
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            outputs = model.generate(**inputs, max_new_tokens=384, temperature=0.1, do_sample=False)
            raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug("LLM raw output (first 300 chars): %s", raw[:300])

            start_idx = raw.find("{")
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(raw)):
                    if raw[i] == "{":
                        brace_count += 1
                    elif raw[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = raw[start_idx : i + 1]
                            llm_result = json.loads(json_str)

                            template_phrases = [
                                "extracted name here", "skill1", "skill2",
                                "degree1", "summary of their work",
                            ]
                            is_template = any(
                                p.lower() in str(llm_result).lower() for p in template_phrases
                            )

                            if not is_template:
                                logger.info("LLM extraction succeeded.")
                                if llm_result.get("name", "").strip():
                                    llm_name = str(llm_result["name"]).strip()
                                    # enforce name quality; fallback to header heuristic if model returns headings/titles
                                    if _is_plausible_person_name(llm_name):
                                        name = _strip_contact_noise(llm_name)
                                if llm_result.get("email", "").strip():
                                    email = llm_result["email"]
                                if llm_result.get("phone", "").strip():
                                    phone = llm_result["phone"]
                                if isinstance(llm_result.get("location"), str) and llm_result.get("location", "").strip():
                                    location = llm_result["location"].strip()
                                if llm_result.get("skills"):
                                    skills = llm_result["skills"]
                                if llm_result.get("education"):
                                    education = llm_result["education"]
                                if llm_result.get("projects"):
                                    projects = llm_result["projects"]
                                if llm_result.get("experience_summary", "").strip():
                                    experience_summary = llm_result["experience_summary"]
                                if llm_result.get("experience_years") is not None:
                                    try:
                                        experience_years = float(llm_result["experience_years"])
                                    except Exception:
                                        pass
                                work_exp = llm_result.get("work_experience")
                                if isinstance(work_exp, list) and work_exp:
                                    # Keep structured work history for robust experience calculation later
                                    work_experience = [j for j in work_exp if isinstance(j, dict)]
                                    try:
                                        yrs = calculate_experience_years(work_exp)
                                        if yrs > 0:
                                            experience_years = yrs
                                            if not experience_summary:
                                                experience_summary = f"{yrs} years experience (from job history)"
                                    except Exception as exc:
                                        logger.debug("work_experience years calc failed: %s", exc)
                                if llm_result.get("summary", "").strip():
                                    summary = llm_result["summary"].strip()
                                if isinstance(llm_result.get("companies_worked_at"), list):
                                    companies_worked_at = [
                                        str(x).strip()
                                        for x in llm_result["companies_worked_at"]
                                        if x
                                    ]
                                if isinstance(llm_result.get("current_role"), str):
                                    current_role = llm_result["current_role"].strip()
                                if isinstance(llm_result.get("important_keywords"), list):
                                    important_keywords = [
                                        str(x).strip()
                                        for x in llm_result["important_keywords"]
                                        if x
                                    ]
                                if isinstance(llm_result.get("key_skills"), list) and llm_result.get("key_skills"):
                                    key_skills = [
                                        str(x).strip()
                                        for x in llm_result["key_skills"]
                                        if x and str(x).strip()
                                    ]
                            else:
                                logger.debug("LLM returned template placeholder, using regex only.")
                            break
        except Exception as exc:
            logger.warning("LLM extraction failed (%s), falling back to regex.", exc)

    # --- experience_summary regex fallback ---
    if not experience_summary:
        m = re.search(r"(\d+(?:\.\d+)?)(?:\+|-)?[\s-]*years?", resume_text.lower())
        if m:
            experience_summary = f"{m.group(0)} experience"

    # --- experience_years fallback ---
    # 1) Prefer conservative professional experience from structured work_experience (excluding internships)
    total_years, experience_level, internship_present, exp_notes = calculate_professional_experience(
        work_experience,
        resume_text,
    )
    if total_years > 0:
        experience_years = total_years
        if not experience_summary:
            experience_summary = f"{round(total_years, 2)} years experience (calculated from work history)"
    else:
        # 2) Fall back to explicit "X years" summary, but keep conservative
        if experience_summary:
            experience_years = calculate_experience_years(experience_summary)
        # 3) Fall back to date ranges in experience section (internships excluded)
        if experience_years == 0.0:
            approx = _estimate_years_from_raw_dates(resume_text)
            if approx > 0:
                experience_years = approx
                if not experience_summary:
                    experience_summary = f"{approx} years experience (from date ranges)"

    if experience_years == 0.0:
        if "experience_unknown" not in extraction_warnings:
            extraction_warnings.append("experience_unknown")

    # Expose additional experience metadata for callers/UI (kept optional in API schema)
    total_experience_years = float(experience_years or 0.0)
    internship_present_flag = bool(internship_present)
    # If no work_exp list was available, internship_present might still be inferred from text in calculate_professional_experience notes
    experience_level_final = experience_level if experience_level else _experience_level_from_years(total_experience_years, internship_only=False)
    experience_notes_final = exp_notes or ""

    # --- Skills validation (prevent hallucinations) ---
    # Re-parse explicit SKILLS/TECH SKILLS section (same sanitised text as earlier pass).
    try:
        section_skills = extract_skills_from_text(base_text)
    except Exception:
        section_skills = []

    # Track which skills came directly from explicit SKILLS sections so we can
    # treat them as high-confidence (less strict support checks).
    section_skill_keys: set[str] = set()
    for s in section_skills or []:
        raw = _strip_skill_decorators(str(s)).strip()
        if not raw:
            continue
        section_skill_keys.add(raw.lower())
        section_skill_keys.add(_norm_header(raw))
        cn = canonicalise_skill(raw)
        if cn:
            section_skill_keys.add(cn.lower())
            section_skill_keys.add(_norm_header(cn))

    merged_skills: list[str] = []
    seen_sk: set[str] = set()
    # Prefer section order, then add regex/LLM/detected skills not already present.
    skill_source_list: list = list(
        dict.fromkeys(
            [str(x).strip() for x in (section_skills or []) if str(x).strip()]
            + [str(x).strip() for x in (skills or []) if str(x).strip()]
        )
    )
    for raw in skill_source_list:
        text_raw = _strip_skill_decorators(str(raw))
        # Filter 1: Drop entries that clearly belong to education (degrees, universities, etc.)
        if _is_education_like_phrase(text_raw):
            continue
        # Filter 2: Drop percentage metrics (e.g., "Studies 60%", "Management 75%")
        if _is_percentage_metric(text_raw):
            continue
        norm = canonicalise_skill(text_raw)
        if not norm:
            continue
        norm = _strip_footnote_numbers(norm)
        if not _skill_token_is_valid(norm):
            continue
        if _skill_is_noise(norm):
            continue
        k = norm.lower()
        if k in seen_sk:
            continue
        tr = text_raw.strip()
        c_tr = canonicalise_skill(tr)
        from_explicit_section = (
            tr.lower() in section_skill_keys
            or _norm_header(tr) in section_skill_keys
            or (bool(c_tr) and c_tr.lower() in section_skill_keys)
        )
        # Skills coming from an explicit SKILLS/TECHNICAL SKILLS section are
        # trusted more and do not require a strict global text support check.
        if from_explicit_section or _skill_is_supported_by_text(norm, resume_text_sanitised or resume_text):
            seen_sk.add(k)
            merged_skills.append(norm)

    skills = merged_skills

    # key_skills are a subset; validate the same way
    merged_key: list[str] = []
    seen_ks: set[str] = set()
    for raw in key_skills or []:
        raw = _strip_skill_decorators(str(raw))
        if _is_education_like_phrase(raw):
            continue
        if _is_percentage_metric(raw):
            continue
        norm = canonicalise_skill(raw)
        if not norm:
            continue
        norm = _strip_footnote_numbers(norm)
        if not _skill_token_is_valid(norm):
            continue
        if _skill_is_noise(norm):
            continue
        k = norm.lower()
        if k in seen_ks:
            continue
        if _skill_is_supported_by_text(norm, resume_text_sanitised or resume_text) or k in seen_sk:
            seen_ks.add(k)
            merged_key.append(norm)
    key_skills = merged_key

    # Evidence-gated filtering + internal weighting (not exposed)
    try:
        filtered_skills, skill_weights = _compute_skill_weights(
            skills or [],
            resume_text=resume_text_sanitised or resume_text,
            section_skills=section_skills or [],
            extraction_low_confidence=extraction_low_confidence,
        )
        if filtered_skills:
            skills = filtered_skills
        # Rebuild key_skills as the top weighted skills, preserving any existing key_skills preference
        if skills:
            key_pref = [canonicalise_skill(s).lower() for s in (key_skills or []) if canonicalise_skill(s)]
            ordered = sorted(
                skills,
                key=lambda s: (
                    1 if s.lower() in set(key_pref) else 0,
                    skill_weights.get(s.lower(), 0),
                    _count_skill_occurrences(s, (resume_text_sanitised or resume_text).lower()),
                ),
                reverse=True,
            )
            key_skills = ordered[:15]
    except Exception as exc:
        logger.debug("skill weighting failed: %s", exc)

    # Internal risk flags (surfaced only via extraction_warnings logs)
    if len(skills or []) >= 60 and "tool_dumping" not in extraction_warnings:
        extraction_warnings.append("tool_dumping")
    if extraction_low_confidence and "low_text_quality" not in extraction_warnings:
        extraction_warnings.append("low_text_quality")

    # --- Boost domain stack (e.g. .NET, React) based on role/experience text ---
    skills, key_skills = _boost_domain_skills(
        resume_text=resume_text,
        skills=skills,
        key_skills=key_skills,
        current_role=current_role,
        experience_summary=experience_summary,
    )

    # Keep only technical-looking skills (vocabulary + tech-shaped tokens), not soft skills.
    skills = [s for s in (skills or []) if _passes_technical_skill_output(str(s), _tech_vocab)]
    key_skills = [s for s in (key_skills or []) if _passes_technical_skill_output(str(s), _tech_vocab)]
    important_keywords = [
        k for k in (important_keywords or []) if _passes_technical_skill_output(str(k), _tech_vocab)
    ]

    # --- Primary vs other skills classification ---
    # Primary skills must be skill names (not designations).
    def _is_primary_candidate_skill(s: str) -> bool:
        if not s:
            return False
        if _skill_is_noise(s):
            return False
        if len(s) < 2 or len(s) > 40:
            return False
        if len(s.split()) > 5:
            return False
        # Prefer core tech labels and common tech punctuation
        return _is_core_tech_label(s) or any(ch in s for ch in (".", "#", "+"))

    # other_skills: cleaned + deduped technical skills list (no connectors/stopwords)
    other_skills = []
    seen_o: set[str] = set()
    for s in skills or []:
        ss = _normalize_skill_token(str(s))
        if not ss or _skill_is_noise(ss):
            continue
        k = ss.lower()
        if k in seen_o:
            continue
        seen_o.add(k)
        other_skills.append(ss)

    # primary_skills: top 3 evidence-weighted skills, excluding junk phrases
    primary_skills = []
    for s in key_skills or []:
        ss = _normalize_skill_token(str(s))
        if not ss or not _is_primary_candidate_skill(ss):
            continue
        if ss.lower() in {x.lower() for x in primary_skills}:
            continue
        primary_skills.append(ss)
        if len(primary_skills) >= 3:
            break
    if not primary_skills:
        # fallback to first few other_skills
        primary_skills = other_skills[:3]

    # Fallback: derive a best-effort name from email local-part when header/LLM failed
    if (not name or name == "Unknown") and email:
        local = email.split("@", 1)[0]
        # Replace common separators with spaces and title-case
        local = re.sub(r"[._\-]+", " ", local)
        local = re.sub(r"\s+", " ", local).strip()
        if local and any(ch.isalpha() for ch in local):
            parts = [p for p in local.split(" ") if p]
            if 1 <= len(parts) <= 4:
                name = " ".join(w.capitalize() for w in parts)
                if "name_low_confidence" not in extraction_warnings:
                    extraction_warnings.append("name_low_confidence")

    logger.info(
        "Extracted: name=%s email=%s skills_count=%d primary=%s exp_years=%.1f",
        name,
        email,
        len(skills),
        ", ".join(primary_skills),
        experience_years,
    )
    try:
        logger.info(
            "extraction_audit %s",
            json.dumps(
                {
                    "extraction_confidence": round(float(extraction_confidence), 3),
                    "low_confidence": bool(extraction_low_confidence),
                    "warnings": extraction_warnings[:10],
                    "skills_count": int(len(skills)),
                    "primary_count": int(len(primary_skills)),
                    "exp_years": float(experience_years or 0.0),
                    **(_conf_meta or {}),
                },
                ensure_ascii=False,
            ),
        )
    except Exception:
        pass

    # Final sanitization before returning/persisting
    summary = _collapse_whitespace(summary or "")
    experience_summary = _collapse_whitespace(experience_summary or "")
    current_role = _collapse_whitespace(current_role or "")
    location = _collapse_whitespace(location or "")

    # Validation/repair contract (runs on normalized text)
    repaired, added = validate_and_repair_extraction(
        {
            "name": name,
            "email": email,
            "phone": phone,
            "location": location,
            "skills": skills or [],
        },
        base_text,
    )
    if added:
        for w in added:
            if w not in extraction_warnings:
                extraction_warnings.append(w)
    name = repaired.get("name") or name
    location = repaired.get("location") or location

    # Strip mistaken "skills" that are actually the candidate's name (e.g. "Gourav" alone).
    _name_toks = _name_tokens_for_skill_exclusion(name)
    if _name_toks:
        skills = [s for s in (skills or []) if not _skill_is_excluded_name_token(s, _name_toks)]
        key_skills = [s for s in (key_skills or []) if not _skill_is_excluded_name_token(s, _name_toks)]
        primary_skills = [s for s in (primary_skills or []) if not _skill_is_excluded_name_token(s, _name_toks)]
        other_skills = [s for s in (other_skills or []) if not _skill_is_excluded_name_token(s, _name_toks)]
    if not primary_skills and skills:
        primary_skills = [s for s in skills if _is_core_tech_label(str(s))][:3]

    return {
        "name": name if name and name != "Unknown" else "Unknown Candidate",
        "email": email,
        "phone": phone,
        "location": location,
        "skills": skills or [],
        "experience_years": float(experience_years) if experience_years else 0.0,
        "total_experience_years": float(total_experience_years) if total_experience_years else 0.0,
        "experience_level": experience_level_final or "",
        "internship_present": bool(internship_present_flag),
        "experience_notes": experience_notes_final or "",
        "experience_summary": experience_summary or "",
        "education": education or [],
        "projects": projects or [],
        "summary": summary,
        "companies_worked_at": companies_worked_at or [],
        "current_role": current_role,
        "important_keywords": important_keywords or [],
        "key_skills": key_skills if key_skills else (skills[:15] if skills else []),
        "primary_skills": primary_skills,
        "other_skills": other_skills,
        "extraction_warnings": extraction_warnings,
    }


# ---------------------------------------------------------------------------
# Enrichment — single LLM call that replaces:
#   generate_summary_and_one_liner()  +  generate_experience_line_and_tags()
# This cuts upload LLM inference from 3 passes → 2 passes.
# ---------------------------------------------------------------------------

def generate_enrichment(extracted: dict, resume_text: str) -> dict:
    """
    Returns a dict with keys:
      summary, one_liner, experience_line, experience_tags
    Uses a single NuExtract inference instead of two separate calls.
    Falls back gracefully on failure.
    """
    skills = extracted.get("skills") or []
    years = extracted.get("experience_years") or 0
    experience_summary = extracted.get("experience_summary") or ""
    context = (
        f"Name: {extracted.get('name', '')}\n"
        f"Current role: {extracted.get('current_role', '')}\n"
        f"Experience years: {years}\n"
        f"Experience summary: {experience_summary}\n"
        f"Skills: {', '.join(skills[:15])}\n"
        f"Companies: {', '.join((extracted.get('companies_worked_at') or [])[:5])}\n"
        f"Education: {', '.join((extracted.get('education') or [])[:4])}\n"
        f"Resume snippet:\n{resume_text[:1000]}"
    )
    prompt = (
        "You are an HR assistant. Based on the candidate data below, provide:\n"
        '1) summary: 2–3 sentences describing the candidate\'s profile.\n'
        '2) one_liner: A single short line like "Senior .NET developer with 5 years in fintech".\n'
        '3) experience_line: A single concise sentence summarising experience, years and main tech.\n'
        '4) experience_tags: 5–12 key technology/domain tags.\n\n'
        f"Candidate data:\n{context}\n\n"
        "Return ONLY a JSON object with exactly:\n"
        '{ "summary": "...", "one_liner": "...", "experience_line": "...", '
        '"experience_tags": ["Tag1", "Tag2", ...] }'
    )

    fallback = _enrichment_fallback(extracted)

    try:
        tokenizer = _get_extract_tokenizer()
        model = _get_extract_model()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.2, do_sample=True)
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

        start_idx = raw.find("{")
        if start_idx != -1:
            brace_count = 0
            for i in range(start_idx, len(raw)):
                if raw[i] == "{":
                    brace_count += 1
                elif raw[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        obj = json.loads(raw[start_idx : i + 1])
                        tags_raw = obj.get("experience_tags") or []
                        if isinstance(tags_raw, str):
                            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
                        elif isinstance(tags_raw, list):
                            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
                        else:
                            tags = fallback["experience_tags"]

                        return {
                            "summary": (obj.get("summary") or fallback["summary"]).strip()[:2000],
                            "one_liner": (obj.get("one_liner") or fallback["one_liner"]).strip()[:500],
                            "experience_line": (obj.get("experience_line") or fallback["experience_line"]).strip()[:500],
                            "experience_tags": tags[:20] or fallback["experience_tags"],
                        }
    except Exception as exc:
        logger.warning("generate_enrichment failed (%s), using fallback.", exc)

    return fallback


def _enrichment_fallback(extracted: dict) -> dict:
    role = extracted.get("current_role") or "Professional"
    years = extracted.get("experience_years") or 0
    skills = (extracted.get("skills") or [])[:5]
    skills_str = ", ".join(skills) if skills else "N/A"
    summary = f"{role} with {years} years of experience. Key skills: {skills_str}."
    one_liner = f"{role} with {years} years experience" + (
        f" in {', '.join(skills[:2])}" if skills else ""
    )
    years_text = format_experience_duration(years)
    exp_line = (
        f"{years_text} of experience" + (f" with {', '.join(skills[:3])}" if skills else "")
        if years_text != "Not specified"
        else (extracted.get("experience_summary") or "")
    )
    return {
        "summary": summary[:2000],
        "one_liner": one_liner[:500],
        "experience_line": exp_line[:500],
        "experience_tags": skills[:10],
    }


# ---------------------------------------------------------------------------
# Legacy wrappers kept for backwards compatibility
# ---------------------------------------------------------------------------

def generate_summary_and_one_liner(extracted: dict, resume_text: str) -> tuple[str, str]:
    """Kept for compatibility. Prefer generate_enrichment() for new code."""
    enrichment = generate_enrichment(extracted, resume_text)
    return enrichment["summary"], enrichment["one_liner"]


def generate_experience_line_and_tags(extracted: dict, resume_text: str) -> tuple[str, list[str]]:
    """Kept for compatibility. Prefer generate_enrichment() for new code."""
    enrichment = generate_enrichment(extracted, resume_text)
    return enrichment["experience_line"], enrichment["experience_tags"]


# ---------------------------------------------------------------------------
# Chat / ranking helpers
# ---------------------------------------------------------------------------

def _run_chat(messages: list[dict], max_new_tokens: int = 250) -> str:
    """
    Send a messages list to the chat pipeline and return the assistant reply.
    Works with TinyLlama (default) and any HuggingFace chat model.
    Falls back to raw text prompt if the model doesn't support the chat template.
    """
    pipe = _get_chat_pipe()
    try:
        out = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False,
                   return_full_text=False)
        result = out[0]["generated_text"]
        # Chat models return either a string or a list of message dicts
        if isinstance(result, list):
            for msg in reversed(result):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg["content"].strip()
            return str(result).strip()
        return str(result).strip()
    except Exception:
        # Fallback: flatten messages to plain text prompt
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ) + "\nASSISTANT:"
        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False,
                   return_full_text=False)
        return out[0]["generated_text"].strip()


def chatbot_answer(question: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise resume Q&A assistant. "
                "Answer ONLY from the provided resume context. "
                "If the answer is not in the context, reply: "
                "Answer not found in the provided data. "
                "Never reveal emails or phone numbers unless the user explicitly asks."
            ),
        },
        {
            "role": "user",
            "content": f"Resume Context:\n{context[:3000]}\n\nQuestion: {question}",
        },
    ]
    return _run_chat(messages, max_new_tokens=250)


def rank_candidates(job_description: str, resumes_context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an HR assistant. Rank candidates from best to worst fit "
                "for the given job description. "
                "Each candidate is shown as '--- Candidate N: REAL_NAME ---'. "
                "You MUST use the candidate's REAL NAME (not 'Candidate N') in your output. "
                "Format each line exactly as: rank, real_name, score/100, one-line reason. "
                "Example: 1, John Smith, 85, Strong Python backend experience."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Job Description:\n{job_description[:2000]}\n\n"
                f"Candidates:\n{resumes_context[:6000]}"
            ),
        },
    ]
    return _run_chat(messages, max_new_tokens=500)


def analyze_fit(job_description: str, candidate_context: str) -> dict:
    """
    Return matched_skills, missing_skills, fit_summary, score_1_10.

    IMPORTANT: Keep output shape stable for the existing UI/API.
    This function now uses a deterministic rubric for the numeric score so it is
    less misleading than an uncalibrated LLM number, while optionally using the
    chat model to improve phrasing of the summary.
    """

    def _extract_years(text: str) -> float:
        if not text:
            return 0.0
        m = re.search(r"\bexperience\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*years?\b", text.lower())
        if not m:
            m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*years?\b", text.lower())
        try:
            return float(m.group(1)) if m else 0.0
        except Exception:
            return 0.0

    def _tokenise(text: str) -> set[str]:
        if not text:
            return set()
        t = text.lower().replace("–", "-")
        # normalise common aliases into simpler tokens
        t = t.replace("node js", "node.js").replace("nodejs", "node.js")
        t = t.replace("dot net", ".net").replace("asp net", "asp.net")
        t = t.replace("k8s", "kubernetes")
        # keep dots/pluses/hashes, split on other punctuation/whitespace
        parts = re.split(r"[^a-z0-9.+#]+", t)
        parts = [p for p in parts if p]
        return set(parts)

    def _extract_tech_terms(text: str) -> set[str]:
        """
        Extract a conservative set of tech terms from free text.
        Uses _CORE_TECH_PRIMARY as the base vocabulary plus a few multiword aliases.
        """
        if not text:
            return set()

        low = text.lower()
        tokens = _tokenise(low)

        # Multiword patterns (must be checked against raw text)
        multi = {
            "google cloud": "gcp",
            "amazon web services": "aws",
            "microsoft azure": "azure",
            "postgre sql": "postgresql",
            "postgres": "postgresql",
            "node.js": "node.js",
            "asp.net": "asp.net",
            ".net": ".net",
        }
        out: set[str] = set()

        for phrase, canon in multi.items():
            if phrase in low:
                out.add(canon)

        for kw in _CORE_TECH_PRIMARY:
            k = kw.lower()
            if k in tokens or k in low:
                out.add(k)

        # Remove non-signal methodology labels if present
        out.discard("scrum")
        out.discard("agile")
        out.discard("kanban")
        return out

    jd = (job_description or "").strip()
    cand = (candidate_context or "").strip()
    if not jd or not cand:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "fit_summary": "Could not compute fit (missing job description or candidate data).",
            "score_1_10": 5,
        }

    jd_terms = _extract_tech_terms(jd)
    cand_terms = _extract_tech_terms(cand)

    matched = sorted(jd_terms.intersection(cand_terms))
    missing = sorted(jd_terms.difference(cand_terms))

    # --- deterministic rubric for score_1_10 ---
    years = _extract_years(cand)
    denom = max(1, len(jd_terms))
    coverage = len(matched) / denom

    # Base score is 4 with coverage contributing up to +5 points.
    score = 4.0 + (5.0 * coverage)

    # Confidence penalties: if we can't extract much signal, cap the score.
    if len(cand_terms) <= 2:
        score = min(score, 6.0)

    # Seniority calibration (very lightweight; avoids obvious overrating).
    jd_low = jd.lower()
    is_senior_jd = any(w in jd_low for w in ("senior", "lead", "staff", "principal", "manager"))
    is_junior_jd = any(w in jd_low for w in ("junior", "intern", "entry", "fresher", "associate"))
    if is_senior_jd and years and years < 4:
        score -= 1.5
    if is_junior_jd and years >= 8:
        score -= 0.5

    # Missing-signal penalty when JD has multiple terms and coverage is low.
    if len(jd_terms) >= 5 and coverage < 0.3:
        score -= 1.0

    score_i = int(round(min(10.0, max(1.0, score))))

    # --- summary: deterministic baseline + optional LLM phrasing ---
    strengths = ", ".join(matched[:6]) if matched else "No clear overlaps detected"
    gaps = ", ".join(missing[:6]) if missing else "No major gaps detected"
    baseline = (
        f"Fit score reflects skills overlap and evidence confidence. "
        f"Strengths: {strengths}. "
        f"Gaps: {gaps}."
    ).strip()

    fit_summary = baseline
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an HR assistant writing a short, high-signal fit note. "
                    "Do NOT invent details. Use the provided matched/missing terms and years. "
                    "Return plain text only (no JSON). 2-3 short sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Job description (truncated):\n{jd[:1200]}\n\n"
                    f"Candidate years: {years}\n"
                    f"Matched terms: {', '.join(matched[:10])}\n"
                    f"Missing terms: {', '.join(missing[:10])}\n\n"
                    f"Write the fit summary."
                ),
            },
        ]
        txt = _run_chat(messages, max_new_tokens=180)
        if isinstance(txt, str) and txt.strip():
            fit_summary = txt.strip()[:900]
    except Exception as exc:
        logger.debug("analyze_fit LLM phrasing failed: %s", exc)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "fit_summary": fit_summary,
        "score_1_10": score_i,
    }


# ---------------------------------------------------------------------------
# Embedding model singleton
# ---------------------------------------------------------------------------
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model…")
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        logger.info("Embedding model ready.")
    return _embedding_model


# ---------------------------------------------------------------------------
# Date / experience helpers
# ---------------------------------------------------------------------------

def parse_date(date_str: str):
    if not date_str:
        return None
    date_str = date_str.lower().strip()
    # Remove common trailing noise
    date_str = date_str.split("|", 1)[0].strip()
    date_str = date_str.split("(", 1)[0].strip()
    date_str = date_str.replace("–", "-").replace("’", "'")
    date_str = re.sub(r"\s+", " ", date_str).strip()

    if any(x in date_str for x in ("present", "current", "till date", "till now", "ongoing")):
        return datetime.today()

    # Normalise month spellings that datetime doesn't accept
    # e.g. "Sept 2024" → "Sep 2024"
    date_str = re.sub(r"\bsept\b", "sep", date_str)

    for fmt in ("%b %Y", "%B %Y", "%Y", "%m/%Y", "%m-%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            # Guardrails: ignore obviously invalid/future dates
            now = datetime.today()
            if dt.year < 1970:
                return None
            if dt > (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=31)):
                return None
            return dt
        except ValueError:
            pass
    return None


def calculate_experience_years(work_experience) -> float:
    """Accept a list of job dicts or a free-form summary string."""
    if isinstance(work_experience, str):
        m = re.search(r"(\d+(?:\.\d+)?)(?:\+|-)?[\s-]*years?", work_experience.lower())
        try:
            return float(m.group(1)) if m else 0.0
        except Exception:
            return 0.0

    periods = []
    for job in work_experience:
        if not isinstance(job, dict):
            continue
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        if start and end and end > start:
            periods.append((start, end))

    if not periods:
        return 0.0

    periods.sort(key=lambda x: x[0])
    merged = [periods[0]]
    for cur in periods[1:]:
        last = merged[-1]
        if cur[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], cur[1]))
        else:
            merged.append(cur)

    total_days = sum((e - s).days for s, e in merged)
    return round(total_days / 365, 2)


_INTERN_KEYWORDS = {"intern", "internship", "trainee", "apprentice"}


def _intern_label(role_text: str) -> str | None:
    """Return normalized internship label if role looks like internship/trainee."""
    if not role_text:
        return None
    s = re.sub(r"\s+", " ", str(role_text).strip().lower())
    if any(k in s for k in ("intern", "internship")):
        return "Intern"
    if any(k in s for k in ("trainee", "apprentice")):
        return "Trainee"
    return None


def _experience_level_from_years(years: float, internship_only: bool) -> str:
    if internship_only:
        return "Fresher with internship experience"
    y = float(years or 0.0)
    if y <= 1:
        return "Fresher / Entry-level"
    if y <= 3:
        return "Junior"
    if y <= 6:
        return "Mid-level"
    return "Senior"


def calculate_professional_experience(
    work_experience: list[dict] | None,
    resume_text: str,
) -> tuple[float, str, bool, str]:
    """
    Conservative experience calculator.
    - Excludes internships/trainee/apprentice roles from professional experience.
    - Uses only roles with parseable dates.
    Returns: (total_years, experience_level, internship_present, notes)
    """
    notes: list[str] = []
    internship_present = False

    jobs = []
    internship_only = False

    if isinstance(work_experience, list) and work_experience:
        for job in work_experience:
            if not isinstance(job, dict):
                continue
            role = str(job.get("role") or "").strip()
            label = _intern_label(role)
            if label:
                internship_present = True
                # Label internships clearly
                job["role"] = label
                continue
            start = parse_date(job.get("start_date"))
            end = parse_date(job.get("end_date"))
            if start and end and end > start:
                jobs.append({"start_date": job.get("start_date"), "end_date": job.get("end_date")})
            else:
                # Missing/unclear dates for a full-time role → be conservative
                if role:
                    notes.append("Some roles have missing/unclear dates; experience may be incomplete.")

        if not jobs and internship_present:
            internship_only = True
            years = 0.0
            lvl = _experience_level_from_years(years, internship_only=True)
            return years, lvl, True, "Fresher with internship experience"

        years = calculate_experience_years(jobs)
        if years <= 0 and notes:
            notes.insert(0, "Experience cannot be reliably calculated")
        lvl = _experience_level_from_years(years, internship_only=False)
        return years, lvl, internship_present, "; ".join(dict.fromkeys([n for n in notes if n]).keys())

    # No structured work_experience → fallback to date ranges in EXPERIENCE section
    approx = _estimate_years_from_raw_dates(resume_text)
    if approx <= 0:
        # If resume mentions internships but has no parseable full-time dates
        if re.search(r"\b(intern|internship|trainee|apprentice)\b", (resume_text or "").lower()):
            return 0.0, "Fresher with internship experience", True, "Fresher with internship experience"
        return 0.0, "Experience cannot be reliably calculated", False, "Experience cannot be reliably calculated"

    lvl = _experience_level_from_years(approx, internship_only=False)
    return approx, lvl, internship_present, ""


def _extract_experience_section(resume_text: str) -> str:
    """
    Return only the text under EXPERIENCE / PROFESSIONAL EXPERIENCE headers.
    This prevents education date ranges (e.g. 2020–2023 BBA) from being
    counted as job experience.
    """
    if not resume_text:
        return ""
    lines = resume_text.splitlines()
    start_idx = None
    end_idx = None
    for i, ln in enumerate(lines):
        norm = _norm_header(ln)
        if norm in {
            "experience",
            "work experience",
            "work experiences",
            "work history",
            "professional experience",
            "employment history",
            "professional summary and experience",
        }:
            start_idx = i + 1
            break
    if start_idx is None:
        return ""
    for j in range(start_idx, len(lines)):
        if _looks_like_header(lines[j]):
            end_idx = j
            break
    if end_idx is None:
        end_idx = len(lines)
    section = "\n".join(lines[start_idx:end_idx]).strip()
    return section


def _estimate_years_from_raw_dates(resume_text: str) -> float:
    """
    Fallback: scan raw resume text for date ranges, but ONLY inside the
    experience section. This avoids counting education years.
    """
    if not resume_text:
        return 0.0

    exp_text = _extract_experience_section(resume_text)
    if not exp_text:
        return 0.0

    month = r"(?:jan|feb|mar|apr|may|jun|june|jul|july|aug|sep|sept|oct|nov|dec)[a-z]*"
    end_word = r"(?:present|current|till date|till now|ongoing)"
    sep = r"(?:–|-|to)"

    patterns = [
        rf"(\d{{1,2}}[/-]\d{{4}})\s*{sep}\s*(\d{{1,2}}[/-]\d{{4}}|{end_word})",
        rf"({month}\s+\d{{4}})\s*{sep}\s*({month}\s+\d{{4}}|{end_word})",
        rf"(\d{{4}})\s*{sep}\s*(\d{{4}}|{end_word})",
    ]

    jobs = []
    text = exp_text.replace("–", "-")
    exp_lines = [ln.strip() for ln in text.splitlines()]

    def _nearby_mentions_intern(match_start: int) -> bool:
        # Map match position to an approximate line index
        prefix = text[:match_start]
        line_idx = prefix.count("\n")
        for k in range(max(0, line_idx - 2), min(len(exp_lines), line_idx + 3)):
            if any(w in (exp_lines[k] or "").lower() for w in _INTERN_KEYWORDS):
                return True
        return False

    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            # Exclude internships/trainee periods conservatively
            if _nearby_mentions_intern(m.start()):
                continue
            s = (m.group(1) or "").strip()
            e = (m.group(2) or "").strip()
            if s and e:
                jobs.append({"start_date": s, "end_date": e})

    if not jobs:
        return 0.0
    return calculate_experience_years(jobs)


def estimate_experience_years_from_text(resume_text: str) -> float:
    """
    Fast, deterministic experience estimator used when the resume doesn't
    explicitly state 'X years'. Avoids calling the LLM.
    """
    if not resume_text:
        return 0.0
    # Prefer date-ranges inside EXPERIENCE section first (more reliable than summary blurbs)
    approx = _estimate_years_from_raw_dates(resume_text)
    if approx > 0:
        return approx

    # Fallback: explicit numeric "X years" anywhere
    m = re.search(r"(\d+(?:\.\d+)?)(?:\+|-)?[\s-]*years?", resume_text.lower())
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return 0.0


def format_experience_duration(years: float) -> str:
    """Return '3 years 4 months' or 'Not specified'."""
    try:
        y = float(years or 0.0)
    except Exception:
        y = 0.0
    if y <= 0:
        return "Not specified"
    total_months = int(round(y * 12))
    yrs, mos = divmod(total_months, 12)
    parts: list[str] = []
    if yrs:
        parts.append(f"{yrs} year{'s' if yrs != 1 else ''}")
    if mos:
        parts.append(f"{mos} month{'s' if mos != 1 else ''}")
    return " ".join(parts)
