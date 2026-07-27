"""
resume_parser_v2.py
===================
Production-Grade Resume Parsing Upgrade Module
Extends (does NOT replace) the existing backend/main.py pipeline.

Features:
  1. Multi-Strategy PDF Extraction  — PyMuPDF (fitz) fallback
  2. Advanced Preprocessing Layer   — remove non-ASCII, icons, duplicates
  3. Resume Type Detection          — technical vs non_technical
  4. Adaptive Skill Engine          — TECH + BUSINESS skill dictionaries
  5. Per-Field Confidence Scoring   — name / skills / overall
  6. Structured Final Output        — with domain and confidence block
  7. Debug Logging throughout

Usage from api.py or main.py:
  from backend.resume_parser_v2 import enrich_extraction_result, extract_text_robust
"""

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. MULTI-STRATEGY PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_with_pymupdf(path: str) -> str:
    """
    PyMuPDF (fitz) extraction — best for multi-column and icon-heavy PDFs.
    Returns empty string if fitz is not installed or fails.
    """
    try:
        import fitz  # PyMuPDF
        text_parts = []
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
        return "\n".join(text_parts).strip()
    except ImportError:
        logger.debug("PyMuPDF (fitz) not installed — skipping fitz extraction.")
        return ""
    except Exception as e:
        logger.debug(f"PyMuPDF extraction failed: {e}")
        return ""


def _text_quality_score(text: str) -> float:
    """
    Score extracted text quality (0–1).
    Higher = better structured, fewer junk chars, more word density.
    """
    if not text or not text.strip():
        return 0.0

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    total_chars = len(text)
    word_count = len(text.split())

    # Word density (words per char)
    word_density = min(1.0, word_count / max(1, total_chars / 5))

    # Non-ASCII ratio (lower = better)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = non_ascii / max(1, total_chars)
    ascii_score = max(0.0, 1.0 - (non_ascii_ratio * 3.0))

    # Line structure (many lines = better PDF parse)
    line_score = min(1.0, len(lines) / 80.0)

    # Keyword presence bonus
    low = text.lower()
    keyword_bonus = 0.0
    for kw in ("experience", "skills", "education", "projects", "summary"):
        if kw in low:
            keyword_bonus += 0.04
    keyword_bonus = min(0.20, keyword_bonus)

    score = (
        0.30 * word_density
        + 0.30 * ascii_score
        + 0.20 * line_score
        + 0.20 * keyword_bonus
    )
    return round(min(1.0, score), 4)


def extract_text_robust(path: str) -> tuple[str, dict]:
    """
    Multi-strategy PDF extraction with automatic arbitration.
    Tries pdfplumber, pypdf, and PyMuPDF (fitz).
    Returns the best version based on quality scoring.

    Returns:
        (best_text, meta_dict)
    """
    import pdfplumber
    from pypdf import PdfReader

    candidates: list[tuple[str, str]] = []  # (source_name, text)

    # Strategy 1: pdfplumber
    try:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            candidates.append(("pdfplumber", text.strip()))
    except Exception as e:
        logger.debug(f"pdfplumber failed: {e}")

    # Strategy 2: pypdf
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        text = "\n".join(parts).strip()
        if text:
            candidates.append(("pypdf", text))
    except Exception as e:
        logger.debug(f"pypdf failed: {e}")

    # Strategy 3: PyMuPDF (fitz) — best for complex/icon-heavy PDFs
    fitz_text = _extract_with_pymupdf(path)
    if fitz_text:
        candidates.append(("pymupdf", fitz_text))

    if not candidates:
        logger.warning(f"All PDF extractors failed for {path}")
        return "", {"error": "all_extractors_failed", "source": None}

    # Score and pick the best
    scored = [(name, text, _text_quality_score(text)) for name, text in candidates]
    scored.sort(key=lambda x: x[2], reverse=True)
    best_name, best_text, best_score = scored[0]

    meta = {
        "source": best_name,
        "quality_score": best_score,
        "candidates": [{"source": n, "score": s} for n, _, s in scored],
    }
    logger.info(
        f"[PDF Extraction] Best source: {best_name} (score={best_score:.3f}) "
        f"from {len(candidates)} strategies"
    )
    return best_text, meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING LAYER
# ─────────────────────────────────────────────────────────────────────────────

# Icon/symbol ranges commonly found in fancy resume templates
_ICON_RANGES = [
    (0xF000, 0xF8FF),   # Private Use Area (FontAwesome icons, etc.)
    (0x1F300, 0x1FAFF), # Emoji ranges
    (0xE000, 0xEFFF),   # More private use
]

def _is_icon_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ICON_RANGES)


def preprocess_resume_text(text: str, max_chars: int = 8000) -> str:
    """
    Advanced preprocessing pipeline:
      1. Strip null bytes
      2. Remove icon/emoji characters
      3. Normalize Unicode (NFKC)
      4. Remove non-ASCII characters
      5. Remove duplicate lines
      6. Strip footers/headers (short repeated lines)
      7. Fix broken spacing
      8. Limit to max_chars

    Returns cleaned text string.
    """
    if not text:
        return ""

    # Step 1: Strip null bytes
    text = text.replace("\u0000", "")

    # Step 2: Remove icon/emoji characters
    text = "".join(ch for ch in text if not _is_icon_char(ch))

    # Step 3: Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Step 4: Replace non-ASCII with space (preserve structure)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Step 5: Normalize line breaks and whitespace per line, applying multi-column formatting
    try:
        try:
            from backend.extraction_v3 import normalize_multi_column
        except ImportError:
            from extraction_v3 import normalize_multi_column
        text = normalize_multi_column(text)
    except Exception as e:
        logger.warning(f"Multi-column normalization failed: {e}")

    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        lines.append(ln)

    # Step 6: Remove duplicate lines (footers/headers) but preserve structure
    seen_lines: dict[str, int] = {}
    for ln in lines:
        if not ln:
            continue
        key = ln.lower().strip()
        seen_lines[key] = seen_lines.get(key, 0) + 1
    # Remove lines that repeat 3+ times (they are headers/footers)
    freq_threshold = 3
    deduped = []
    for ln in lines:
        if not ln:
            deduped.append("")
            continue
        key = ln.lower().strip()
        if seen_lines.get(key, 0) < freq_threshold:
            deduped.append(ln)
    lines = deduped

    # Step 7: Fix broken spacing (e.g. "Java Script" → "JavaScript")
    result = "\n".join(lines).strip()
    result = re.sub(r"(?i)\bjava\s+script\b", "JavaScript", result)
    result = re.sub(r"(?i)\btype\s+script\b", "TypeScript", result)
    result = re.sub(r"(?i)\bnode\s+\.?\s*js\b", "Node.js", result)

    # Step 8: Limit to max_chars
    if len(result) > max_chars:
        result = result[:max_chars]
        logger.debug(f"[Preprocess] Text truncated to {max_chars} chars")

    logger.info(
        f"[Preprocess] Input: {len(text)} chars → Output: {len(result)} chars "
        f"({len(lines)} lines)"
    )
    return result.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESUME TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_TECH_KEYWORDS = {
    "python", "java", "javascript", "typescript", "c#", "c++", "php", "golang",
    "ruby", "react", "angular", "vue", "node.js", "django", "flask", "spring",
    ".net", "asp.net", "sql", "mysql", "postgresql", "mongodb", "nosql", "redis",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux", "api",
    "microservices", "restful", "graphql", "html", "css", "bootstrap", "jquery",
    "android", "ios", "swift", "kotlin", "flutter", "react native",
    "tensorflow", "pytorch", "machine learning", "deep learning", "nlp",
    "opencv", "opencv", "data science", "scikit-learn", "pandas", "numpy",
    "uipath", "rpa", "power bi", "tableau", "hadoop", "spark", "kafka",
    "elasticsearch", "devops", "ci/cd", "jenkins", "terraform", "ansible",
    "fastapi", "express", "laravel", "rails", "hibernate", "spring boot",
}

_BUSINESS_KEYWORDS = {
    "sales", "marketing", "crm", "erp", "tally", "quickbooks", "excel", "word",
    "powerpoint", "communication", "leadership", "negotiation", "customer service",
    "business development", "account management", "project management", "vendor",
    "procurement", "supply chain", "operations", "hr", "recruitment", "payroll",
    "finance", "accounting", "budget", "forecasting", "strategy", "branding",
    "social media", "content", "seo", "digital marketing", "email marketing",
    "client relations", "stakeholder", "cross-functional", "reporting",
    "team management", "people management", "training", "compliance",
}


def detect_resume_type(text: str) -> dict:
    """
    Classify the resume as 'technical' or 'non_technical'.
    Returns: {"type": "technical"|"non_technical", "tech_score": int, "biz_score": int}
    """
    low = (text or "").lower()
    tech_hits = sum(1 for kw in _TECH_KEYWORDS if kw in low)
    biz_hits = sum(1 for kw in _BUSINESS_KEYWORDS if kw in low)

    resume_type = "technical" if tech_hits >= 3 or (tech_hits > biz_hits) else "non_technical"
    logger.info(
        f"[Resume Type] tech_score={tech_hits}, biz_score={biz_hits} → {resume_type}"
    )
    return {
        "type": resume_type,
        "tech_score": tech_hits,
        "biz_score": biz_hits,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUSINESS SKILL DICTIONARY (for non-technical resumes)
# ─────────────────────────────────────────────────────────────────────────────

BUSINESS_SKILLS_VOCAB = {
    # Core Business
    "excel", "word", "powerpoint", "outlook", "tally", "quickbooks", "sap", "erp", "crm",
    "salesforce", "zoho", "hubspot", "ms office",
    # Communication & Management
    "communication", "leadership", "team management", "people management",
    "negotiation", "presentation", "public speaking", "stakeholder management",
    "conflict resolution", "decision making", "problem solving",
    # Sales & Marketing
    "sales", "business development", "account management", "customer service",
    "client relations", "marketing", "digital marketing", "seo", "sem",
    "social media marketing", "content marketing", "email marketing", "branding",
    "lead generation", "market research", "competitor analysis",
    # Operations & HR
    "operations", "supply chain", "procurement", "vendor management", "logistics",
    "hr", "recruitment", "talent acquisition", "payroll", "onboarding",
    "performance management", "training and development", "compensation",
    # Finance & Accounting
    "finance", "accounting", "budgeting", "forecasting", "financial analysis",
    "cost reduction", "p&l", "balance sheet", "gst", "taxation", "audit",
    # Project Management
    "project management", "agile", "scrum", "kanban", "pmp",
    "risk management", "cross-functional", "reporting", "kpi tracking",
    # Data (Non-technical)
    "data analysis", "data entry", "mis reporting", "excel dashboards", "pivot tables",
    "vlookup", "power bi", "tableau", "google analytics",
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONFIDENCE SCORING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def compute_name_confidence(
    name: str,
    source: str,  # 'ner', 'header', 'email', 'unknown'
    score: float,
) -> float:
    """
    Compute name extraction confidence (0–1).
    Based on source reliability and scoring strength.
    """
    if not name or name in {"Unknown", "Unknown Candidate"}:
        return 0.0

    # Base confidence by source
    base = {
        "ner": 0.85,
        "header": 0.70,
        "header_combined": 0.60,
        "email": 0.40,
        "unknown": 0.20,
    }.get(source, 0.30)

    # Bonus: 2-3 word names are more reliable
    words = name.strip().split()
    if 2 <= len(words) <= 3:
        base += 0.10
    elif len(words) == 1:
        base -= 0.15

    # Clamp to [0, 1]
    return round(max(0.0, min(1.0, base + min(score * 0.1, 0.1))), 3)


def compute_skills_confidence(
    primary_skills: list,
    other_skills: list,
    max_score: float,
    resume_type: str,
) -> float:
    """
    Compute skill extraction confidence (0–1).
    """
    if primary_skills == ["Nil"] or not primary_skills:
        return 0.0

    total_skills = len(primary_skills) + len(other_skills)
    # More skills found = higher confidence (up to a point)
    count_score = min(1.0, total_skills / 10.0)

    # Strength of top score
    score_strength = min(1.0, max_score / 15.0)

    # Bonus if we found skills from multiple sources
    multi_source = 1.0 if total_skills >= 3 else 0.6

    # Penalty if resume is non-technical but we're using tech skills
    type_penalty = 0.0
    if resume_type == "non_technical" and len(primary_skills) > 0 and primary_skills[0] != "Nil":
        type_penalty = 0.1  # slight penalty as confidence is lower for non-tech

    conf = (0.40 * count_score + 0.40 * score_strength + 0.20 * multi_source) - type_penalty
    return round(max(0.0, min(1.0, conf)), 3)


def compute_overall_confidence(
    name_conf: float,
    skills_conf: float,
    text_quality: float,
) -> float:
    """
    Weighted overall confidence across all fields.
    """
    overall = (
        0.35 * name_conf
        + 0.40 * skills_conf
        + 0.25 * text_quality
    )
    return round(max(0.0, min(1.0, overall)), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 6. DOMAIN DETECTION (Enhanced)
# ─────────────────────────────────────────────────────────────────────────────

_ENHANCED_DOMAIN_MAP = {
    "Computer Vision": {
        "keywords": ["opencv", "mediapipe", "yolo", "image processing", "object detection",
                     "facial recognition", "video analytics"],
        "skills_boost": ["PyTorch", "TensorFlow", "OpenCV", "MediaPipe", "NumPy"],
    },
    "RPA": {
        "keywords": ["uipath", "automation anywhere", "blue prism", "rpa", "robotic process"],
        "skills_boost": ["UiPath", "Python", "Automation Anywhere", "Blue Prism"],
    },
    "Data Analyst": {
        "keywords": ["sql", "power bi", "tableau", "excel", "data analysis", "mis", "reporting", "pandas"],
        "skills_boost": ["SQL", "Power BI", "Tableau", "Excel", "Pandas", "NumPy"],
    },
    "Machine Learning / AI": {
        "keywords": ["machine learning", "deep learning", "neural network", "nlp",
                     "natural language", "llm", "large language model", "bert", "gpt"],
        "skills_boost": ["TensorFlow", "PyTorch", "Scikit-learn", "Python", "Keras"],
    },
    "DevOps / Cloud": {
        "keywords": ["devops", "ci/cd", "docker", "kubernetes", "terraform", "ansible",
                     "jenkins", "aws", "azure", "gcp", "cloud"],
        "skills_boost": ["Docker", "Kubernetes", "AWS", "Azure", "GCP", "Jenkins", "Terraform"],
    },
    "Full Stack Development": {
        "keywords": ["full stack", "frontend", "backend", "react", "angular", "node.js", "api"],
        "skills_boost": ["React", "Node.js", "JavaScript", "TypeScript", "HTML", "CSS"],
    },
    "Business / Operations": {
        "keywords": ["operations", "supply chain", "procurement", "erp", "sap", "crm",
                     "vendor management", "logistics"],
        "skills_boost": ["SAP", "ERP", "CRM", "Excel", "PowerPoint"],
    },
    "Sales / Marketing": {
        "keywords": ["sales", "business development", "marketing", "seo", "digital marketing",
                     "lead generation", "client relations"],
        "skills_boost": ["Salesforce", "HubSpot", "Excel", "PowerPoint", "Google Analytics"],
    },
}


def detect_domain_enhanced(text: str) -> tuple[str, list[str]]:
    """
    Detect the candidate's primary domain and return skills to boost.
    Returns: (domain_name, list_of_skills_to_boost)
    """
    low = (text or "").lower()
    domain_scores: dict[str, int] = {}

    for domain, config in _ENHANCED_DOMAIN_MAP.items():
        hits = sum(1 for kw in config["keywords"] if kw in low)
        if hits > 0:
            domain_scores[domain] = hits

    if not domain_scores:
        return "General", []

    best_domain = max(domain_scores, key=lambda d: domain_scores[d])
    boost_skills = _ENHANCED_DOMAIN_MAP[best_domain]["skills_boost"]

    logger.info(
        f"[Domain Detection] Detected: '{best_domain}' "
        f"(score={domain_scores[best_domain]}, "
        f"all_scores={dict(sorted(domain_scores.items(), key=lambda x: -x[1])[:3])})"
    )
    return best_domain, boost_skills


# ─────────────────────────────────────────────────────────────────────────────
# 7. ADAPTIVE SKILL EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def filter_skills_by_vocab(
    skills: list[str],
    resume_type: str,
    tech_vocab: set,
) -> list[str]:
    """
    Adaptively filter skills based on resume type.
    - Technical resumes: keep only tech vocabulary
    - Non-technical: also accept business skill vocabulary
    """
    result = []
    for s in skills:
        s_low = s.lower().strip()
        if s_low == "nil":
            result.append(s)
            continue
        # Always check tech vocab
        if s_low in tech_vocab or s_low in {v.lower() for v in BUSINESS_SKILLS_VOCAB}:
            result.append(s)
            continue
        # For non-technical, be more lenient
        if resume_type == "non_technical" and len(s_low) >= 3:
            # Accept anything that's in business vocab case-insensitively
            if any(bv.lower() == s_low for bv in BUSINESS_SKILLS_VOCAB):
                result.append(s)
    return result


def boost_domain_skills_v2(
    primary_skills: list[str],
    other_skills: list[str],
    boost_skills: list[str],
) -> tuple[list[str], list[str]]:
    """
    Promote domain-relevant skills to the front of primary_skills.
    """
    if not boost_skills:
        return primary_skills, other_skills

    existing_lower = {s.lower() for s in primary_skills}
    for skill in boost_skills:
        if skill.lower() in existing_lower:
            # Already in primary, move to front
            for i, s in enumerate(primary_skills):
                if s.lower() == skill.lower():
                    primary_skills.insert(0, primary_skills.pop(i))
                    break
        elif any(s.lower() == skill.lower() for s in other_skills):
            # Promote from other_skills to primary
            for i, s in enumerate(other_skills):
                if s.lower() == skill.lower():
                    primary_skills.insert(0, other_skills.pop(i))
                    existing_lower.add(skill.lower())
                    break

    return primary_skills, other_skills


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN ENRICHMENT FUNCTION (calls into existing pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_extraction_result(
    raw_extracted: dict,
    resume_text: str,
    text_quality_score: float = 0.5,
    name_source: str = "unknown",
    name_score: float = 0.0,
) -> dict:
    """
    Primary enrichment function.
    Takes the result dict from extract_resume() (backend/main.py) and enhances it:

    1. Detects resume type (technical / non-technical)
    2. Performs enhanced domain detection
    3. Boosts domain-relevant skills
    4. Computes per-field confidence scores
    5. Returns the final enriched output dict

    Args:
        raw_extracted: dict from extract_resume()
        resume_text:   full preprocessed resume text
        text_quality_score: score from extract_text_robust() (0–1)
        name_source:   how the name was found ('ner', 'header', 'email', etc.)
        name_score:    raw score from select_best_name() (0–10 range approx)

    Returns:
        Enriched dict with confidence block and domain field.
    """

    # ── Resume Type Detection ──────────────────────────────────────────────
    type_info = detect_resume_type(resume_text)
    resume_type = type_info["type"]

    # ── Domain Detection ───────────────────────────────────────────────────
    domain, boost_skills = detect_domain_enhanced(resume_text)

    # ── Skill Boost ────────────────────────────────────────────────────────
    primary = list(raw_extracted.get("primary_skills") or [])
    other = list(raw_extracted.get("other_skills") or [])

    if primary != ["Nil"]:
        primary, other = boost_domain_skills_v2(primary, other, boost_skills)

    # ── Confidence Scoring ─────────────────────────────────────────────────
    name = raw_extracted.get("name", "")
    name_conf = compute_name_confidence(name, name_source, name_score)

    # Max skill score — estimate from the number and diversity of skills
    total_skills = len(primary) + len(other)
    estimated_max_score = min(15.0, total_skills * 2.5) if primary != ["Nil"] else 0.0
    skills_conf = compute_skills_confidence(primary, other, estimated_max_score, resume_type)

    overall_conf = compute_overall_confidence(name_conf, skills_conf, text_quality_score)

    # ── Debug Logging ──────────────────────────────────────────────────────
    logger.info(
        f"[Enrichment] name='{name}' (conf={name_conf:.2f}, source={name_source}) | "
        f"type={resume_type} | domain={domain} | "
        f"primary_skills={primary[:3]} | skills_conf={skills_conf:.2f} | "
        f"overall_conf={overall_conf:.2f}"
    )

    # ── Compose Final Output ───────────────────────────────────────────────
    return {
        "name": name,
        "email": raw_extracted.get("email", ""),
        "phone": raw_extracted.get("phone", ""),
        "location": raw_extracted.get("location", ""),
        "primary_skills": primary,
        "secondary_skills": other,
        "skills": primary + other if primary != ["Nil"] else ["Nil"],
        "experience_years": raw_extracted.get("experience_years", 0.0),
        "total_experience_years": raw_extracted.get("total_experience_years", 0.0),
        "experience_level": raw_extracted.get("experience_level", ""),
        "experience_summary": raw_extracted.get("experience_summary", ""),
        "experience_notes": raw_extracted.get("experience_notes", ""),
        "education": raw_extracted.get("education", []),
        "projects": raw_extracted.get("projects", []),
        "summary": raw_extracted.get("summary", ""),
        "companies_worked_at": raw_extracted.get("companies_worked_at", []),
        "current_role": raw_extracted.get("current_role", ""),
        "important_keywords": raw_extracted.get("important_keywords", []),
        "key_skills": primary[:15],
        "other_skills": other,
        "extraction_warnings": raw_extracted.get("extraction_warnings", []),
        "resume_type": resume_type,
        "domain": domain,
        "confidence": {
            "name": name_conf,
            "skills": skills_conf,
            "overall": overall_conf,
            "text_quality": round(text_quality_score, 3),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. FAIL-SAFE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def safe_enrich(raw_extracted: dict, resume_text: str, **kwargs) -> dict:
    """
    Fail-safe wrapper around enrich_extraction_result.
    If enrichment fails for any reason, returns the original dict
    with default confidence values and a warning.
    """
    try:
        return enrich_extraction_result(raw_extracted, resume_text, **kwargs)
    except Exception as e:
        logger.error(f"[Enrichment] safe_enrich failed: {e}")
        # Return original with minimal additions
        return {
            **raw_extracted,
            "resume_type": "unknown",
            "domain": "General",
            "confidence": {
                "name": 0.0,
                "skills": 0.0,
                "overall": 0.0,
                "text_quality": 0.0,
            },
        }
