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
            # Core skills headers
            "skills", "technical skills", "tech skills", "key skills",
            "core competencies", "competencies", "skill set", "skillset",
            # Technology/Stack headers
            "tech stack", "technology stack", "technologies", "technical stack",
            "tools", "tools and technologies", "tools technologies", "tools & technologies",
            "programming languages", "languages", "programming",
            # Expertise headers
            "expertise", "technical expertise", "areas of expertise",
            "professional skills", "technical skills",
            # Proficiency headers
            "proficiencies", "technical proficiencies", "technical capabilities",
            "capabilities", "proficiency level",
            # Knowledge headers
            "technical knowledge", "knowledge", "domain knowledge",
            # Certifications (sometimes grouped with skills)
            "certifications", "certificates", "licenses",
            # Platform-specific
            "platforms", "frameworks", "libraries", "packages", "sdks",
            # Cloud/DevOps
            "cloud", "cloud platforms", "devops tools",
            # Generic variations
            "core strengths", "it skills", "technical skill set",
            "skills summary", "key competencies", "technical competencies",
            "relevant skills", "professional expertise", "hr skills", "key hr skills",
            "management skills", "core technical skills",
            # Additional variations
            "technical", "soft skills", "hard skills", "key technical skills",
            "software skills", "computer skills", "it proficiency",
            "technical abilities", "core skills", "main skills",
        }
        if norm in known:
            return True
        if "skills" in norm or "competencies" in norm or "proficiencies" in norm or "technologies" in norm:
            if not any(bad in norm for bad in ("experience", "education", "history", "objective", "summary", "profile", "projects", "work", "employment")):
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
    Priority-based extraction with improved edge case handling:
    1. First 3-5 lines
    2. Regex for names (expanded patterns with special characters)
    3. Reject emails, company names, titles, phone numbers
    4. Handle hyphens, apostrophes, and mixed case
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    top_lines = lines[:5]
    
    reject_patterns = [
        r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", # Email
        r"(?i)\b(?:pvt|ltd|limited|inc|llc|corporation|technologies|solutions|technology|informatics|systems|services|group|consulting|labs|global|international|india|pvt\.ltd)\b", # Companies
        r"\d{4,}", # Phone/Numbers
        r"(?i)\b(developer|engineer|manager|curriculum vitae|resume|profile|summary|analyst|executive|architect|consultant|specialist|lead|director|officer|coordinator|administrator|senior|junior|assistant)\b", # Roles/Headings
        r"(?i)\b(objective|career goal|professional summary|about me|contact|phone|email|address|portfolio|github|linkedin)\b", # Section headers
        r"https?://[^\s]+", # URLs
        r"(?i)\b(page|cv|vitae)\b", # Document type
    ]
    
    # Expanded name regex patterns with support for special characters
    name_patterns = [
        r"^[A-Z][a-z]+(?:[-'\s][A-Z][a-z]+){1,2}$",  # Standard with hyphens/apostrophes: John Doe, Mary-Jane Smith, O'Connor
        r"^[A-Z][a-z]+(?:[-'\s][A-Z]\.?\s*){1,2}[A-Z][a-z]+$",  # With initials: J. A. Doe, J. A Doe
        r"^[A-Z]\.?\s+[A-Z][a-z]+(?:[-'\s][A-Z]\.?\s*)?[A-Z][a-z]+$",  # Initial first: J. John Doe
        r"^[A-Z][a-z]+(?:[-'\s][A-Z][a-z]+){1,3}$",  # Longer names (up to 4 words): Mary Jane Smith Johnson
        r"^[A-Z][a-z]+(?:[-'\s][A-Z][a-z]+){0,1}$",  # Single or double word: John, John Doe
    ]
    
    for line in top_lines:
        # Preserve hyphens and apostrophes in names
        line_clean = re.sub(r"[^a-zA-Z\s\-']", "", line).strip()
        
        rejected = False
        for pattern in reject_patterns:
            if re.search(pattern, line):
                rejected = True
                break
                
        if rejected:
            continue
            
        # Try all name patterns
        for name_regex in name_patterns:
            if re.match(name_regex, line_clean):
                # Additional validation: must have at least 2 parts or be a valid single name
                words = [w for w in re.split(r"[-'\s]+", line_clean) if w]
                if len(words) >= 2:
                    return line_clean
                elif len(words) == 1 and len(words[0]) >= 2:
                    # Single word name - validate it's not a common word
                    if words[0].lower() not in {"the", "and", "for", "with", "from", "name"}:
                        return line_clean
        
        # Handle ALL CAPS names
        words = [w for w in re.split(r"[-'\s]+", line_clean) if w]
        if line_clean.isupper() and 2 <= len(words) <= 4:
            return line_clean.title()
        
        # Handle Title Case names with special characters
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            return line_clean
        
        # Handle names with mixed case (e.g., "McDonald", "MacDonald")
        if 2 <= len(words) <= 4:
            # Check if it looks like a name (starts with capital, contains letters)
            if all(re.match(r"^[A-Z][a-z]*$", w) for w in words):
                return line_clean
    
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
    Section-first skill extraction with flexible pattern matching.
    If a dedicated skills section is found, explicitly extracts everything listed in it.
    Otherwise, falls back to vocabulary-matching tech keywords in the full text.
    Also uses pattern-based extraction to capture skills not in vocabulary.
    """
    extracted_skills = []
    weak_phrases = {"familiar with", "basic knowledge", "exposure to", "worked on", "understanding of"}
    tech_vocab_lower = {v.lower() for v in (tech_vocab or set())}
    
    def extract_by_patterns(text: str) -> list[str]:
        """Extract skills using common patterns even if not in vocabulary."""
        skills = []
        lines = text.splitlines()
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 2:
                continue
            
            # Skip section headers (more lenient check)
            line_low = line.lower()
            header_keywords = ("skills", "technologies", "tools", "experience", "education", "projects", 
                              "summary", "objective", "profile", "contact", "phone", "email", "address")
            if any(h in line_low for h in header_keywords):
                # Only skip if it's a short line (likely a header)
                if len(line) < 50 and not any(c in line for c in [",", "|", "•", "-"]):
                    continue
            
            # Pattern 1: Comma-separated list (very lenient)
            if "," in line:
                for part in line.split(","):
                    part = part.strip()
                    # Lower threshold: 1-60 chars, check for letters
                    if 1 <= len(part) <= 60 and re.search(r"[a-zA-Z]", part):
                        # Remove common weak phrases
                        if not any(phrase in part.lower() for phrase in weak_phrases):
                            # Remove trailing punctuation
                            part = re.sub(r"[.,;:!\?\)\]\}]+$", "", part).strip()
                            if part:
                                skills.append(part)
            
            # Pattern 2: Bullet points (very lenient)
            elif line.startswith(("-", "•", "*", "+", ">", "o")) or re.match(r"^\d+[\).\]]", line):
                skill = re.sub(r"^[-•*+\d.\)\]>o\s]+", "", line).strip()
                if 1 <= len(skill) <= 60 and re.search(r"[a-zA-Z]", skill):
                    if not any(phrase in skill.lower() for phrase in weak_phrases):
                        skill = re.sub(r"[.,;:!\?\)\]\}]+$", "", skill).strip()
                        if skill:
                            skills.append(skill)
            
            # Pattern 3: Colon-separated (Category: skill1, skill2)
            elif ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    skills_part = parts[1].strip()
                    for skill in re.split(r",\s*|\s+\|\s+", skills_part):
                        skill = skill.strip()
                        if 1 <= len(skill) <= 60 and re.search(r"[a-zA-Z]", skill):
                            if not any(phrase in skill.lower() for phrase in weak_phrases):
                                skill = re.sub(r"[.,;:!\?\)\]\}]+$", "", skill).strip()
                                if skill:
                                    skills.append(skill)
            
            # Pattern 4: Pipe-separated (skill1 | skill2 | skill3)
            elif "|" in line and line.count("|") >= 2:
                for skill in line.split("|"):
                    skill = skill.strip()
                    if 1 <= len(skill) <= 60 and re.search(r"[a-zA-Z]", skill):
                        if not any(phrase in skill.lower() for phrase in weak_phrases):
                            skill = re.sub(r"[.,;:!\?\)\]\}]+$", "", skill).strip()
                            if skill:
                                skills.append(skill)
        
        return skills
    
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
            # Also try pattern-based extraction for skills not in vocabulary
            extracted_skills.extend(extract_by_patterns(skills_section))
    else:
        # No dedicated skills section found - fallback to vocab matching in full text
        # and pattern-based extraction
        extracted_skills.extend(process_text_vocab(full_text))
        extracted_skills.extend(extract_by_patterns(full_text))
        
    # Format & Deduplicate with spelling normalization
    final_skills = []
    seen = set()
    for s in extracted_skills:
        # Normalize spelling first
        normalized = _normalize_skill_spelling(s)
        
        # If it's a short technical acronym, uppercase it; otherwise, title-case or preserve case
        if len(normalized) <= 3:
            norm = normalized.upper()
        else:
            # Title case if it doesn't already contain camelCase or mixed case
            norm = normalized.title() if normalized.islower() or normalized.isupper() else normalized
            
        if norm.lower() not in seen:
            seen.add(norm.lower())
            final_skills.append(norm)
            
    return final_skills


def _build_creative_skill_vocab() -> set[str]:
    """
    Expanded vocabulary for design / 3D / media resumes.
    This keeps deterministic extraction useful on non-software resumes
    without depending on an LLM.
    """
    return {
        # 3D & Animation
        "autodesk maya", "maya", "substance painter", "substance 3d painter",
        "arnold renderer", "arnold", "rizom uv", "blender", "3ds max",
        "cinema 4d", "zbrush", "houdini", "unreal engine", "unity 3d",
        # Adobe Creative Suite
        "adobe photoshop", "photoshop", "after effects", "premiere pro",
        "illustrator", "indesign", "adobe xd", "lightroom", "bridge",
        # Design Tools
        "figma", "sketch", "invision", "corel draw", "affinity designer",
        "affinity photo", "procreate", "canva",
        # 3D Skills
        "rendering", "product visualization", "hard surface modeling",
        "3d modeling", "texturing", "uv unwrapping", "rigging", "animation",
        "character modeling", "environment art", "lighting",
        # Video & Motion
        "video editing", "motion graphics", "vfx", "compositing",
        "color grading", "final cut pro", "da vinci resolve",
        # Additional Creative
        "graphic design", "ui design", "ux design", "web design",
        "branding", "typography", "layout design", "digital art",
    }


def _normalize_skill_spelling(skill: str) -> str:
    """
    Normalize skill spelling by fixing common misspellings and variations.
    Returns the corrected skill name.
    """
    skill_lower = skill.lower().strip()
    
    # Common misspellings and variations mapping
    corrections = {
        # Programming Languages
        "javascipt": "JavaScript",
        "javascript": "JavaScript",
        "javascrip": "JavaScript",
        "java script": "JavaScript",
        "typescript": "TypeScript",
        "type script": "TypeScript",
        "phyton": "Python",
        "pyton": "Python",
        "python": "Python",
        "csharp": "C#",
        "c sharp": "C#",
        "c++": "C++",
        "cpp": "C++",
        "c plus plus": "C++",
        # Frameworks & Libraries
        "reactjs": "React",
        "react js": "React",
        "react.js": "React",
        "react": "React",
        "vuejs": "Vue",
        "vue js": "Vue",
        "vue.js": "Vue",
        "vue": "Vue",
        "angularjs": "Angular",
        "angular js": "Angular",
        "angular.js": "Angular",
        "angular": "Angular",
        "nodejs": "Node.js",
        "node js": "Node.js",
        "node.js": "Node.js",
        "node": "Node.js",
        "expressjs": "Express",
        "express js": "Express",
        "express.js": "Express",
        "express": "Express",
        "django": "Django",
        "flask": "Flask",
        "springboot": "Spring Boot",
        "spring boot": "Spring Boot",
        "springboot": "Spring Boot",
        "laravel": "Laravel",
        "rails": "Rails",
        "ruby on rails": "Rails",
        # Databases
        "mysql": "MySQL",
        "postgresql": "PostgreSQL",
        "postgres": "PostgreSQL",
        "mongodb": "MongoDB",
        "mongo": "MongoDB",
        "mssql": "SQL Server",
        "sql server": "SQL Server",
        "sqlite": "SQLite",
        "redis": "Redis",
        # Cloud & DevOps
        "aws": "AWS",
        "amazon web services": "AWS",
        "azure": "Azure",
        "microsoft azure": "Azure",
        "gcp": "GCP",
        "google cloud": "GCP",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "k8s": "Kubernetes",
        "jenkins": "Jenkins",
        "terraform": "Terraform",
        "ansible": "Ansible",
        "git": "Git",
        "github": "GitHub",
        "gitlab": "GitLab",
        # Frontend
        "html": "HTML",
        "css": "CSS",
        "bootstrap": "Bootstrap",
        "tailwind": "Tailwind CSS",
        "tailwind css": "Tailwind CSS",
        "jquery": "jQuery",
        "sass": "SASS",
        "scss": "SCSS",
        # Tools & Software
        "photoshop": "Photoshop",
        "adobe photoshop": "Photoshop",
        "illustrator": "Illustrator",
        "figma": "Figma",
        "sketch": "Sketch",
        "blender": "Blender",
        "maya": "Maya",
        "autodesk maya": "Maya",
        "unity": "Unity",
        "unity 3d": "Unity",
        "unreal": "Unreal Engine",
        "unreal engine": "Unreal Engine",
        # Data Science
        "tensorflow": "TensorFlow",
        "pytorch": "PyTorch",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "scikit learn": "Scikit-learn",
        "scikit-learn": "Scikit-learn",
        "sklearn": "Scikit-learn",
        "matplotlib": "Matplotlib",
        "seaborn": "Seaborn",
        # Other
        "excel": "Excel",
        "microsoft excel": "Excel",
        "word": "Word",
        "microsoft word": "Word",
        "powerpoint": "PowerPoint",
        "microsoft powerpoint": "PowerPoint",
        "outlook": "Outlook",
        "tableau": "Tableau",
        "power bi": "Power BI",
        "powerbi": "Power BI",
        "sap": "SAP",
        "tally": "Tally",
        "salesforce": "Salesforce",
        "jira": "Jira",
        "confluence": "Confluence",
        "slack": "Slack",
        "zoom": "Zoom",
        "teams": "Microsoft Teams",
        "microsoft teams": "Microsoft Teams",
    }
    
    # Check for exact match first
    if skill_lower in corrections:
        return corrections[skill_lower]
    
    # Check for partial match (case-insensitive)
    for misspelling, correct in corrections.items():
        if skill_lower == misspelling:
            return correct
    
    # If no correction found, return original with proper casing
    # Title case for multi-word, uppercase for short acronyms
    if len(skill) <= 3:
        return skill.upper()
    elif skill.isupper():
        return skill.title()
    elif skill.islower():
        return skill.title()
    else:
        return skill


def _is_valid_person_name(name: str) -> bool:
    """
    Validate that a candidate string is a plausible person name.
    Rejects titles, companies, headings, phone numbers.
    """
    if not name or len(name) > 60:
        return False

    BAD_KEYWORDS = {
        "resume", "cv", "profile", "summary", "objective", "cover letter",
        "engineer", "developer", "manager", "analyst", "director", "architect",
        "consultant", "specialist", "lead", "principal", "officer", "executive",
        "inc", "ltd", "corp", "company", "solutions", "technologies", "services",
        "phone", "email", "address", "github", "linkedin", "portfolio",
        "unknown", "anonymous", "n/a", "none"
    }

    name_lower = name.lower().strip()
    if any(bad in name_lower for bad in BAD_KEYWORDS):
        return False

    parts = name.split()
    if len(parts) < 2 and len(parts[0]) < 3:
        return False

    if len(parts) > 4:
        return False

    for part in parts:
        if not re.match(r"^[a-z'\-]+$", part, re.IGNORECASE):
            return False

    if re.search(r"\d{3,}", name):
        return False

    return True


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
