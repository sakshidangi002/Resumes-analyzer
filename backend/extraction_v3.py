import re
from datetime import datetime

logger = __import__('logging').getLogger(__name__)

# ---------------------------------------------------------------------------
# 6. Multi-Column Handling Fix
# ---------------------------------------------------------------------------
def normalize_multi_column(text: str) -> str:
    """
    Normalizes multi-column resumes before extraction by detecting uneven line lengths
    and merging them appropriately.
    """
    if not text:
        return ""
    
    # Try to identify columnar artifacts: lots of spaces separating text on same line.
    lines = text.splitlines()
    normalized_lines = []
    
    for line in lines:
        # If there is a massive gap (e.g. 4+ spaces) indicating columns
        if re.search(r'\s{4,}', line):
            parts = re.split(r'\s{4,}', line)
            # Reconstruct them as separate lines to avoid mixing left/right contexts blindly
            normalized_lines.extend([p.strip() for p in parts if p.strip()])
        else:
            normalized_lines.append(line.strip())
            
    # Remove empty lines but do not aggressively destroy section breaks
    result = []
    blank_count = 0
    for ln in normalized_lines:
        if not ln:
            blank_count += 1
            if blank_count <= 2:
                result.append("")
        else:
            blank_count = 0
            # Fix mid-word spurious spaces (PDF ligature artifact): "Progra mming" -> "Programming"
            # Only join when: lowercase letters [a-z] on both sides of a single space,
            # AND neither fragment is a standalone real word longer than 1 char
            # We target short fragments (< 7 chars) followed by more lowercase chars
            fixed = re.sub(r'([a-z]{2,6}) ([a-z]{2,6})\b', lambda m: _try_join_broken_word(m), ln)
            result.append(fixed)
            
    return "\n".join(result).strip()


def _try_join_broken_word(m: re.Match) -> str:
    """
    Heuristically join two lowercase fragments if they look like a broken word.
    E.g. 'Progra mming' -> we detect 'progra' + 'mming' don't form real words alone
    but together 'programming' is valid. Simple check: reject if either piece ends 
    in a common English word suffix that makes it standalone.
    """
    left = m.group(1)
    right = m.group(2)
    # Common standalone short words to NOT merge
    standalone = {
        "in", "an", "of", "to", "at", "by", "or", "as", "is", "it",
        "be", "on", "up", "no", "so", "do", "go", "my", "we", "us",
        "for", "the", "and", "are", "but", "not", "was", "has", "had",
        "its", "can", "may", "one", "two", "our", "via", "per", "age",
        "data", "open", "free", "code", "core", "base", "word", "work",
        "from", "with", "this", "that", "well", "also", "into", "java",
        "html", "node", "next", "nest", "pipe", "time", "mail", "user",
    }
    if left.lower() in standalone or right.lower() in standalone:
        return m.group(0)  # don't merge standalone words
    # If right fragment starts with a non-word-start pattern (like 'mming', 'tion', 'ing', 'ment')
    broken_suffixes = ('ing', 'ion', 'ment', 'tion', 'ness', 'ance', 'ence', 'ity',
                       'ive', 'ful', 'less', 'ble', 'age', 'ure', 'ous')
    if right.lower().startswith(('mm', 'nd', 'tt', 'ss', 'pp', 'll', 'rr', 'cc')) or \
       right.lower() in [s.lstrip('aeiou') for s in broken_suffixes] or \
       any(right.lower().startswith(suf) for suf in ('mming', 'tion', 'ning', 'ring', 'ling', 'king')):
        return left + right  # merge the fragments
    return m.group(0)  # keep as-is


# ---------------------------------------------------------------------------
# 1. Add Section-Based Parsing (CRITICAL)
# ---------------------------------------------------------------------------
def split_resume_sections(text: str) -> dict:
    """
    Splits resume text into sections using regex + heuristics.
    Returns dictionary with predefined keys: skills, experience, education, projects.
    """
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "projects": ""
    }
    
    # Heuristic section headers mapping
    header_patterns = {
        "experience": r"^(experience|work experience|professional experience|employment history|work history)\b",
        "education": r"^(education|academic background|academics|qualifications)\b",
        "projects": r"^(projects|project experience|personal projects|academic projects)\b"
    }

    def is_skills_header(line_clean):
        norm = re.sub(r"[^a-z0-9 ]+", " ", line_clean.lower()).strip()
        if not norm:
            return False
        known = {
            "skills", "technical skills", "tech skills", "key skills", "core competencies",
            "skill set", "skillset", "tech stack", "technology stack", "technologies",
            "tools", "tools technologies", "tools and technologies", "expertise",
            "technical expertise", "areas of expertise", "professional skills", "key hr skills", "hr skills",
            "core strengths", "it skills", "technical skill set", "skills summary",
            "key competencies", "technical competencies", "relevant skills",
            "technical proficiencies", "proficiencies", "technical capabilities",
        }
        if norm in known:
            return True
        if "skills" in norm or "competencies" in norm or "proficiencies" in norm or "technologies" in norm:
            if not any(bad in norm for bad in ("experience", "education", "history", "objective", "summary", "profile", "projects")):
                return True
        return False

    current_section = None
    section_content = {k: [] for k in sections.keys()}

    lines = text.splitlines()
    for line in lines:
        clean_line = line.strip().lower()
        clean_line = re.sub(r"[^a-z0-9 ]+", " ", clean_line).strip()
        
        # Check if line matches any section header
        matched_section = None
        if len(clean_line) < 40: # headers are usually short
            if is_skills_header(clean_line):
                matched_section = "skills"
            else:
                for section, pattern in header_patterns.items():
                    if re.match(pattern, clean_line):
                        matched_section = section
                        break
        
        if matched_section:
            current_section = matched_section
        elif current_section:
            section_content[current_section].append(line)
            
    # Reassemble and clean up sections
    for section in sections.keys():
        content = "\n".join(section_content[section]).strip()
        sections[section] = content

    return sections


# ---------------------------------------------------------------------------
# 2. Replace LLM-based Experience Calculation
# ---------------------------------------------------------------------------
def extract_experience_years(experience_text: str) -> float:
    """
    Extract date ranges from experience text using regex and compute total duration.
    E.g. extracts 'Jan 2020 - Mar 2023' or '2021 - Present'.
    """
    if not experience_text:
        return 0.0

    total_months = 0
    # Date formats to catch: "Jan 2020 - Mar 2023", "01/2020 - 05/2023", "2020 - Present", "Aug 2018 to Oct 2020"
    date_regex = r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{2,4}|\b\d{1,2}/\d{2,4}|\b\d{4}\b"
    range_regex = rf"({date_regex})\s*(?:-|to|–)\s*({date_regex}|present|current|now|till date)"
    
    matches = re.findall(range_regex, experience_text)
    current_date = datetime.now()
    
    def parse_date(date_str):
        date_str = date_str.lower().strip()
        if date_str in ["present", "current", "now", "till date"]:
            return current_date
            
        formats = ["%b %Y", "%B %Y", "%m/%Y", "%m/%y", "%Y"]
        for fmt in formats:
            try:
                clean_str = re.sub(r'[^a-z0-9\s/]', '', date_str).strip()
                return datetime.strptime(clean_str, fmt)
            except ValueError:
                pass
        return None

    visited_ranges = []
    for start_raw, end_raw in matches:
        start_date = parse_date(start_raw)
        end_date = parse_date(end_raw)
        
        if start_date and end_date and start_date <= end_date:
            # Overlap protection heuristically: if duplicate year/month ranges
            if (start_date, end_date) in visited_ranges:
                continue
            visited_ranges.append((start_date, end_date))
            
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            if 0 < months < 360: # Limit to 30 years
                total_months += months
                
    return round(total_months / 12.0, 1)


# ---------------------------------------------------------------------------
# 3. Fix Name Extraction (Deterministic First)
# ---------------------------------------------------------------------------
def extract_deterministic_name(text: str) -> str:
    """
    Priority-based extraction:
    1. First 3-5 lines
    2. Regex for names
    3. Reject emails, company names, titles
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    top_lines = lines[:5]
    
    reject_patterns = [
        r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", # Email
        r"(?i)\b(?:pvt|ltd|limited|inc|llc|corporation|technologies|solutions|technology|informatics)\b", # Companies
        r"\d{4,}", # Phone/Numbers
        r"(?i)\b(developer|engineer|manager|curriculum vitae|resume|profile|summary|analyst|executive)\b" # Roles/Headings
    ]
    
    name_regex = r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$"
    
    for line in top_lines:
        line_clean = re.sub(r"[^a-zA-Z\s]+", "", line).strip()
        
        rejected = False
        for pattern in reject_patterns:
            if re.search(pattern, line):
                rejected = True
                break
                
        if rejected:
            continue
            
        if re.match(name_regex, line_clean):
            return line_clean
            
        words = line_clean.split()
        if line_clean.isupper() and 2 <= len(words) <= 3:
            return line_clean.title()
            
    return ""


# ---------------------------------------------------------------------------
# 4. Improve Skill Extraction
# ---------------------------------------------------------------------------

# Common English / career-objective / soft-prose tokens that frequently leak
# into a parsed "Skills" section (because the resume layout drifts into a
# summary or objective right after the header). These are NEVER skills, no
# matter how prominently they appear.
_NON_SKILL_PROSE_TOKENS = {
    # Career-goal / objective vocabulary
    "growth", "growth opportunity", "growth opportunities", "opportunity",
    "opportunities", "career", "career growth", "career objective",
    "objective", "objectives", "goal", "goals", "aim", "ambition",
    "looking", "seeking", "interested", "motivation", "motivated",
    "passion", "passionate", "dedicated", "dedication",
    # Filler superlatives / soft fluff
    "good", "great", "excellent", "strong", "quick", "fast", "best",
    "skilled", "skill", "skills", "knowledge", "knowledgeable",
    "etc", "etc.", "and", "or", "the", "with", "of", "in", "on",
    # Soft-skill phrases on their own line
    "team player", "quick learner", "fast learner", "self learner",
    "self motivated", "self-motivated", "hard working", "hardworking",
    "responsible", "result oriented", "result-oriented", "results oriented",
    "results-oriented", "detail oriented", "detail-oriented",
}


def _strip_trailing_punct(s: str) -> str:
    """Remove trailing whitespace + sentence punctuation (. , ; : ! ? –) and outer quotes."""
    return re.sub(r"[\s.,;:!?\u2013\u2014\"'`)]+$", "", s.strip())


def _looks_like_sentence_fragment(original_line: str, cleaned: str) -> bool:
    """A multi-word token that originally ended with sentence punctuation is prose, not a skill."""
    if " " not in cleaned:
        return False
    return bool(re.search(r"[.!?]\s*$", original_line))


def _looks_like_prose_token(cleaned: str, tech_vocab_lower: set) -> bool:
    """
    Reject a candidate skill token if it looks like English prose.
    A token is considered prose when it satisfies any of:
      - Lower-cased form is in the curated NON_SKILL_PROSE_TOKENS blocklist.
      - Starts lowercase and contains multiple words (sentence fragment).
      - Contains > 4 words (too long for a typical skill name).
    Tokens that exist in tech_vocab always survive (e.g. "Node.js").
    """
    if not cleaned:
        return True
    low = cleaned.lower()
    if low in tech_vocab_lower:
        return False
    if low in _NON_SKILL_PROSE_TOKENS:
        return True
    if len(cleaned.split()) > 4:
        return True
    if " " in cleaned and cleaned[0].islower():
        # multi-word fragment starting lowercase, e.g. "looking for growth"
        return True
    return False


def extract_skills_deterministic(skills_section: str, full_text: str, tech_vocab: set) -> list[str]:
    """
    Section-first skill extraction. 
    If a dedicated skills section is found, explicitly extracts everything listed in it.
    Otherwise, falls back to vocabulary-matching tech keywords in the full text.
    """
    extracted_skills = []
    weak_phrases = {"familiar with", "basic knowledge", "exposure to", "worked on", "understanding of"}
    tech_vocab_lower = {v.lower() for v in (tech_vocab or set())}
    
    def process_text_vocab(text):
        words = re.split(r"[,\n|•\u2022;]+", text.lower())
        skills = []
        for word in words:
            word = word.strip()
            if any(phrase in word for phrase in weak_phrases):
                continue
            
            for skill in tech_vocab:
                pattern = rf"(?<![a-z0-9.]){re.escape(skill)}(?![a-z0-9.])"
                if re.search(pattern, word):
                    skills.append(skill)
        return skills

    def process_section_explicitly(section_text: str) -> list[str]:
        # First, join PDF-broken words (e.g., "Progra\nmming" or "Progra mming" -> "Programming")
        # These appear when PDF parsers split a single word across lines or insert spurious spaces.
        # Strategy: join lines that end with a lowercase fragment (no punctuation)
        # Step 1: Rejoin broken hyphenated words (e.g. "Pro-\ngramming")
        section_text = re.sub(r"-\n(\S)", r"\1", section_text)

        raw_lines = section_text.splitlines()
        skills = []

        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line:
                continue

            # Strip leading bullet/dash/star decorators
            line = re.sub(r"^[-•\u2022\uFFFD\s*+►▪]+", "", line).strip()
            if not line:
                continue

            line_low = line.lower()

            # Skip lines that are purely the section header itself
            if any(h == line_low for h in ("skills", "technologies", "proficiencies", "competencies", "tools")):
                continue
            if any(phrase in line_low for phrase in weak_phrases):
                continue

            # Detect "Category Label: skill1, skill2, ..." pattern
            # e.g. "Computer Vision: OpenCV, MediaPipe" or "Programming Languages: Python, HTML"
            colon_match = re.match(r"^([^:]{2,40}):\s*(.+)$", line)
            if colon_match:
                # The part after the colon contains the actual skills
                skills_part = colon_match.group(2).strip()
                # Split the skills part by comma
                for s in re.split(r",\s*", skills_part):
                    s_clean = _strip_trailing_punct(s)
                    if not s_clean or not (2 <= len(s_clean) <= 65):
                        continue
                    if s_clean.lower() in weak_phrases:
                        continue
                    if _looks_like_prose_token(s_clean, tech_vocab_lower):
                        continue
                    skills.append(s_clean)
                continue

            # Plain line - could be "Python, MySQL, Excel" or just "Teamwork"
            # Split on commas only if the line looks like a list
            if "," in line:
                for p in line.split(","):
                    p_clean = _strip_trailing_punct(p)
                    if not p_clean or not (2 <= len(p_clean) <= 65):
                        continue
                    if p_clean.lower() in weak_phrases:
                        continue
                    if _looks_like_prose_token(p_clean, tech_vocab_lower):
                        continue
                    skills.append(p_clean)
            else:
                # Single skill or multi-word skill on its own line.
                # This is the most permissive branch and is what historically let
                # sentence fragments like "Growth." leak through. Apply the
                # strictest filter here:
                #   - strip trailing punctuation
                #   - reject if the original line ended in a sentence terminator
                #     and the cleaned token has multiple words (prose fragment)
                #   - reject prose / soft-fluff tokens
                cleaned = _strip_trailing_punct(line)
                if not cleaned or not (2 <= len(cleaned) <= 65):
                    continue
                if cleaned.lower() in weak_phrases:
                    continue
                if _looks_like_sentence_fragment(line, cleaned):
                    continue
                if _looks_like_prose_token(cleaned, tech_vocab_lower):
                    continue
                skills.append(cleaned)

        return skills
    
    if skills_section and skills_section != full_text:
        # First, try to explicitly extract everything in the skills section
        explicit = process_section_explicitly(skills_section)
        if explicit:
            extracted_skills.extend(explicit)
        else:
            # Fallback to vocab matching inside section only (never full document)
            extracted_skills.extend(process_text_vocab(skills_section))
        
    # Format & Deduplicate
    final_skills = []
    seen = set()
    for s in extracted_skills:
        # If it's a short technical acronym, uppercase it; otherwise, title-case or preserve case
        if len(s) <= 3:
            norm = s.upper()
        else:
            # Title case if it doesn't already contain camelCase or mixed case
            norm = s.title() if s.islower() or s.isupper() else s
            
        if norm.lower() not in seen:
            seen.add(norm.lower())
            final_skills.append(norm)
            
    return final_skills


def _build_creative_skill_vocab() -> set[str]:
    """
    Small add-on vocabulary for design / 3D / media resumes.
    This keeps deterministic extraction useful on non-software resumes
    without depending on an LLM.
    """
    return {
        "autodesk maya",
        "maya",
        "substance painter",
        "substance 3d painter",
        "arnold renderer",
        "arnold",
        "rizom uv",
        "adobe photoshop",
        "photoshop",
        "blender",
        "3ds max",
        "cinema 4d",
        "zbrush",
        "after effects",
        "premiere pro",
        "figma",
        "illustrator",
        "indesign",
        "corel draw",
        "photoshop",
        "rendering",
        "product visualization",
        "hard surface modeling",
        "3d modeling",
        "texturing",
        "uv unwrapping",
    }


def _name_confidence_from_header(text: str, name: str) -> float:
    """High confidence when the name comes from the top of the document."""
    if not text or not name:
        return 0.0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    top = lines[:5]
    name_low = name.strip().lower()
    if any(name_low == ln.lower() for ln in top[:1]):
        return 0.97
    if any(name_low == ln.lower() for ln in top[:3]):
        return 0.92
    return 0.75


def extract_explicit_experience(text: str) -> float:
    if not text:
        return 0.0

    text_low = text.lower()
    month_values: list[int] = []
    seen_months: set[int] = set()

    # 1. Look for month mentions: (\d+)\s*months?
    # Repeated resumes often mention the same duration in both the summary and
    # the role heading. Count each distinct duration only once to avoid
    # inflating "11 months" into "22 months".
    for m in re.finditer(r"(\d+)\s*(?:months?|mos?)\b", text_low):
        val = int(m.group(1))
        if 0 < val < 120 and val not in seen_months:
            seen_months.add(val)
            month_values.append(val)

    if month_values:
        return round(sum(month_values) / 12.0, 2)

    # 2. Look for year mentions: (\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b
    year_values: list[float] = []
    seen_years: set[float] = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", text_low):
        val = float(m.group(1))
        if 0 < val < 40 and val not in seen_years:
            seen_years.add(val)
            year_values.append(val)

    if year_values:
        # Convert distinct year mentions once each. This keeps repeated summary
        # text from inflating the estimate while still allowing multiple
        # different year durations to contribute.
        return round(sum(year_values), 2)

    return 0.0


def deterministic_extract_pipeline(text: str, tech_vocab: set | None = None) -> dict:
    """
    Deterministic resume extraction used as a fallback/override for noisy PDFs.
    Returns a small, stable dict with the fields main.py expects.
    """
    text = normalize_multi_column(text or "")
    sections = split_resume_sections(text)

    name = extract_deterministic_name(text)
    name_conf = _name_confidence_from_header(text, name)

    vocab = set(tech_vocab or set())
    vocab.update(_build_creative_skill_vocab())

    skills_section = sections.get("skills", "") or ""
    # Skills are extracted only from the dedicated skills section (no full-text scan).
    skills = extract_skills_deterministic(skills_section, text, vocab)

    experience_text = sections.get("experience", "") or sections.get("projects", "")
    experience_years = extract_experience_years(experience_text)
    if experience_years == 0.0:
        # Prefer the dedicated experience block so a summary mention and the
        # role heading don't get counted twice. Fall back to the full document
        # only if the section itself doesn't contain an explicit duration.
        experience_years = extract_explicit_experience(experience_text)
    if experience_years == 0.0:
        experience_years = extract_explicit_experience(text)

    return {
        "name": name,
        "name_conf": name_conf,
        "skills": skills,
        "experience_years": experience_years,
    }
