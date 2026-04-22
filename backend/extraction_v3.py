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
            if blank_count <= 2:  # allow up to 2 consecutive blank lines to preserve structure
                result.append("")
        else:
            blank_count = 0
            result.append(ln)
            
    return "\n".join(result).strip()


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
        "skills": r"^(skills|technical skills|technologies|tools\s*&?\s*technologies|core competencies|expertise|tech stack)\b",
        "experience": r"^(experience|work experience|professional experience|employment history|work history)\b",
        "education": r"^(education|academic background|academics|qualifications)\b",
        "projects": r"^(projects|project experience|personal projects|academic projects)\b"
    }

    current_section = None
    section_content = {k: [] for k in sections.keys()}

    lines = text.splitlines()
    for line in lines:
        clean_line = line.strip().lower()
        clean_line = re.sub(r"[^a-z0-9 ]+", " ", clean_line).strip()
        
        # Check if line matches any section header
        matched_section = None
        if len(clean_line) < 40: # headers are usually short
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
def extract_skills_deterministic(skills_section: str, full_text: str, tech_vocab: set) -> list[str]:
    """
    Section-first skill extraction matching against pre-defined vocab.
    Filters out weak phrases (e.g. "familiar with").
    """
    extracted_skills = []
    weak_phrases = {"familiar with", "basic knowledge", "exposure to", "worked on", "understanding of"}
    
    def process_text(text):
        words = re.split(r"[,\n|•\u2022;]+", text.lower())
        skills = []
        for word in words:
            word = word.strip()
            if any(phrase in word for phrase in weak_phrases):
                continue
            
            # Use whole word matching for each tech_vocab skill
            for skill in tech_vocab:
                if len(skill) <= 2:
                    # Strict for C, R, TS -> wait actually use regex word boundaries
                    if re.search(rf"(?<![a-z0-9.]){re.escape(skill)}(?![a-z0-9.])", word):
                        skills.append(skill)
                elif skill in word:
                    # Avoid substring overlap for very short ones, but usually acceptable
                    skills.append(skill)
        return skills
    
    if skills_section:
        extracted_skills.extend(process_text(skills_section))
    
    # Only fallback to full text if section yielded nothing
    if not extracted_skills:
        extracted_skills.extend(process_text(full_text))
        
    # Format identically
    final_skills = []
    seen = set()
    for s in extracted_skills:
        norm = s.title() if len(s)>3 else s.upper() # basic
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
    skills = extract_skills_deterministic(skills_section, text, vocab)
    if not skills:
        # Use the full text as a fallback so resumes without a dedicated SKILLS
        # header still produce a usable tool list.
        skills = extract_skills_deterministic(text, text, vocab)

    experience_text = sections.get("experience", "") or sections.get("projects", "")
    experience_years = extract_experience_years(experience_text)

    return {
        "name": name,
        "name_conf": name_conf,
        "skills": skills,
        "experience_years": experience_years,
    }
