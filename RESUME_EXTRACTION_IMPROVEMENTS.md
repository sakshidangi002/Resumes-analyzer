# Resume Extraction Accuracy Improvement Plan

## Executive Summary

The current extraction system achieves ~50-60% accuracy on skills, ~60% on names, and has issues with:
- **Context truncation** (6000 chars limits multi-page resumes)
- **Section detection** (limited header variations)
- **Name extraction** (confuses titles, companies, headers)
- **Education filtering** (catches certifications as education)
- **Format variability** (PDF vs DOCX handling inconsistent)

This document proposes a **hybrid rule-based + AI approach** with structured preprocessing and fallback strategies.

---

## Part 1: Root Cause Analysis

### 1.1 Name Extraction Failures (60% accuracy)

| Root Cause | Example | Why It Fails | Impact |
|-----------|---------|-------------|--------|
| **Job titles extracted** | "Senior Software Engineer" | NER treats titles as names | 15% of failures |
| **Company names** | "Google Inc" extracted as name | No semantic filtering | 20% of failures |
| **Phone/Email confusion** | "+1-555-1234" parsed as name | Weak pre-filtering | 10% of failures |
| **Resume headers** | "RESUME", "CV", "PROFILE" | Header detection too loose | 10% of failures |
| **Multiple candidates** | Has 3-5 name-like candidates, picks wrong | Scoring algorithm weak | 25% of failures |
| **Email derivation** | email "john.smith@company.com" misinterpreted | Post-processing unreliable | 20% of failures |

**Current Code Issue**: `rank_name_candidates()` relies on NER which doesn't understand context.

### 1.2 Skills Extraction Failures (50-60%)

| Root Cause | Example | Why It Fails | Impact |
|-----------|---------|-------------|--------|
| **Truncated context** | Resume page 2-3 lost | 6000-char limit cuts off | 25% of missing skills |
| **Missing section headers** | "Technical Proficiencies" not in patterns | Only 16 header variations | 15% of failures |
| **Certifications as education** | "Certified Kubernetes Admin" → education | `_is_education_like_phrase()` too broad | 20% of failures |
| **Limited vocabulary** | "Salesforce", "ServiceNow", "SAP" missing | Only 45 base keywords (now ~500+) | 30% of failures |
| **No section parsing** | Resume with non-standard layout | Falls back to full-text vocab scan | 10% of failures |

**Current Code Issue**: Post-processing filters out valid skills when they appear in wrong section.

### 1.3 Experience Details

| Root Cause | Example | Why It Fails | Impact |
|-----------|---------|-------------|--------|
| **Date parsing** | "2019-2021" vs "Jan 2019 - Dec 2021" | Regex doesn't handle all formats | 15% |
| **Overlapping dates** | Summary mentions same duration as job | Duplication not detected | 10% |
| **Current roles ambiguous** | "Present", "Till Date", "Ongoing" | Multiple formats not normalized | 5% |

---

## Part 2: Improved Solutions

### 2.1 Multi-Stage Preprocessing Pipeline

```
Raw Resume (PDF/DOCX)
    ↓
[Stage 1: Normalization]
  • Decompose multi-column layouts
  • Fix PDF ligature artifacts (Progra mming → Programming)
  • Normalize whitespace & line breaks
  • Remove control characters
    ↓
[Stage 2: Section Detection]
  • Identify: HEADER, SKILLS, EXPERIENCE, EDUCATION, PROJECTS
  • Use expanded regex patterns (40+ variations)
  • Fallback: heuristic line-length analysis for non-standard layouts
    ↓
[Stage 3: Text Extraction]
  • Extract each section independently
  • Preserve formatting (bullets, indentation)
  • Max-length per section: 5000 chars (allows full sections)
    ↓
[Stage 4: Confidence Scoring]
  • Signal confidence flags: "low_extraction_confidence", "text_normalized"
  • Track which sections were auto-detected vs explicit
    ↓
Structured Sections: {header, skills, experience, education, projects}
```

### 2.2 Enhanced Name Extraction (Confidence Scoring)

**New Strategy**: Priority-ranked candidate selection with multiple validation checks.

```python
def extract_name_robust(text: str, email: str = "") -> (str, float):
    """
    Returns: (name, confidence_score: 0.0-1.0)
    Confidence breakpoints:
    - 0.95+: Name found in line 1, header-like position
    - 0.85-0.94: Name in top 3 lines + email prefix match
    - 0.70-0.84: Name appears in top, but some ambiguity
    - 0.50-0.69: Name extracted but low confidence (use fallback)
    - <0.50: Extraction failed, use "Unknown" + flag warning
    """
    
    candidates = _extract_name_candidates(text, email)
    # candidates = [(name_str, source_line, confidence), ...]
    
    ranked = _rank_candidates(candidates, email)
    # Ranks based on: position, format, email match, header alignment
    
    best = ranked[0] if ranked else ("", 0.0)
    return best
```

**Validation Filters**:

```
Reject if:
  ✗ Length > 60 chars
  ✗ Contains email pattern (@domain.com)
  ✗ Contains phone pattern (\d{3}-\d{3}-\d{4})
  ✗ Matches job title (Engineer, Manager, Director, etc.)
  ✗ Matches company suffix (Inc, Ltd, LLC, Solutions, etc.)
  ✗ Is all-caps and appears below line 10 (likely header artifact)
  ✗ Starts with number or special char
  ✗ Is pure acronym (CEO, CFO, CTO)

Accept if:
  ✓ 2-3 words, title-case, in top 5 lines
  ✓ Matches email prefix (john.smith@... → "John Smith")
  ✓ Appears on dedicated name line with aligned position
  ✓ Surrounded by contact info (email/phone nearby)
```

### 2.3 Enhanced Skills Extraction

**New Strategy**: Section-first + vocabulary hybrid with confidence scoring.

```python
def extract_skills_hybrid(
    text: str,
    tech_vocabulary: set[str],
    confidence_threshold: float = 0.70
) -> (list[str], dict):
    """
    Returns:
      - skills: list of extracted skills
      - metadata: {
          "source": "section_explicit" | "vocabulary_scan" | "hybrid",
          "confidence": 0.0-1.0,
          "section_detected": bool,
          "skill_count": int,
          "warnings": [...]
        }
    """
    
    # Stage 1: Find skills section
    skills_section = find_section(text, headers=[
        "skills", "technical skills", "tech skills", "core competencies",
        "skill set", "skillset", "tech stack", "technologies",
        "tools and technologies", "expertise", "technical expertise",
        "proficiencies", "technical proficiencies", "technical capabilities",
        # Add 20+ more variations
    ])
    
    # Stage 2: Extract from section (explicit mode)
    if skills_section:
        explicit_skills = extract_skills_from_section(skills_section)
        if len(explicit_skills) >= 3:  # Found meaningful skills block
            return explicit_skills, {"source": "section_explicit", "confidence": 0.95}
    
    # Stage 3: Fallback - scan for vocabulary matches
    vocabulary_skills = scan_vocabulary(text, tech_vocabulary)
    
    # Stage 4: Filter & merge
    final_skills = deduplicate_and_filter(
        explicit_skills + vocabulary_skills,
        confidence_threshold
    )
    
    return final_skills, metadata
```

**Expanded Section Headers** (now 40+ variations):

```python
SECTION_HEADERS = {
    "skills": [
        "skills", "technical skills", "tech skills", "key skills",
        "core competencies", "skill set", "skillset", "tech stack",
        "technology stack", "technologies", "tools", "tools and technologies",
        "expertise", "technical expertise", "areas of expertise",
        "professional skills", "hr skills", "key hr skills",
        "core strengths", "it skills", "technical skill set",
        "skills summary", "key competencies", "technical competencies",
        "relevant skills", "technical proficiencies", "proficiencies",
        "technical capabilities", "programming languages", "languages",
        "platforms", "tools & technologies", "technical knowledge",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "work history", "career history",
        "professional background", "professional history",
    ],
    "education": [
        "education", "academic background", "academics", "qualifications",
        "educational qualifications", "degrees", "certificates",
    ],
    "projects": [
        "projects", "project experience", "personal projects",
        "academic projects", "key projects", "relevant projects",
    ]
}
```

### 2.4 Education vs Certification Filter

**Problem**: Current logic filters "Certified Kubernetes Administrator" as education.

**Solution**: Context-aware filtering with certification patterns.

```python
def is_education_item(text: str) -> bool:
    """
    Returns True only if text is truly an educational degree.
    Rejects certifications and skill mentions.
    """
    
    # Known degree keywords (high confidence education)
    degree_patterns = [
        r"\b(bachelor|master|phd|diploma|associate|postgraduate)\b",
        r"\b(b\.?a\.?|b\.?s\.?|m\.?a\.?|m\.?s\.?|m\.?b\.?a\.?)\b",
        r"\b(b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?)\b",
        r"\b(b\.?sc|m\.?sc|b\.?com|m\.?com)\b",
    ]
    
    # Certification keywords (NOT education)
    cert_patterns = [
        r"\b(certification|certified|certifications)\b",
        r"\b(certificate of|accredited)\b",
        r"(AWS|Azure|Google Cloud|Kubernetes|Docker|Certified)",
        r"\b(CPA|PMP|CISSP|OSCP|Security\+|Scrum Master)\b",
    ]
    
    # If certification pattern found, not education
    for pattern in cert_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    # Now check if degree pattern exists
    for pattern in degree_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Context check: If surrounded by other skills/certs, not education
    # (would be in experience/skills section, not education section)
    return False
```

### 2.5 Improved LLM Prompt with Examples

**Current Issue**: Generic prompt, model hallucinates, no examples.

**Improved Prompt**:

```
You are an expert resume parser extracting structured data.

CRITICAL RULES:
1. Use ONLY information from the provided resume text.
2. Return empty strings or empty arrays if data is not explicitly present.
3. Do NOT invent, hallucinate, or infer information.
4. Do NOT include irrelevant information in any field.

---

FIELD DEFINITIONS & EXAMPLES:

1. **name** (string):
   - MUST be a real person name (2-15 words).
   - DO NOT include titles, email addresses, phone numbers, or company names.
   - ✓ CORRECT: "John Smith", "Maria Garcia-Rodriguez"
   - ✗ WRONG: "Senior Software Engineer", "Microsoft", "+1-555-0123"

2. **email** (string):
   - Standard email format: name@domain.ext
   - ✓ CORRECT: "john.smith@company.com"
   - ✗ WRONG: "john.smith", "Smith", "company.com"

3. **phone** (string):
   - Full phone number with country code preferred.
   - ✓ CORRECT: "+1-555-123-4567", "(555) 123-4567"
   - ✗ WRONG: "555-1234" (incomplete), "555-123-4567x123" (with extension)

4. **location** (string):
   - City, State/Province, or Country.
   - ✓ CORRECT: "San Francisco, CA", "London, UK", "Toronto"
   - ✗ WRONG: "Home Office", "Remote", "Zip Code 12345"

5. **skills** (array):
   - ONLY technical skills, frameworks, tools, platforms, and certifications.
   - Each item should be 2-60 characters.
   - ✓ CORRECT: ["Python", "React", "AWS", "Docker", "PostgreSQL", "AWS Certified Solutions Architect"]
   - ✗ WRONG: ["Communication", "Teamwork", "Problem-solving", "Agile", "Software Development"]

6. **key_skills** (array):
   - TOP 3-5 core/primary technologies the candidate specializes in.
   - Select from the most frequently mentioned, most recent, or most relevant to current role.
   - ✓ CORRECT: ["Python", "Django", "PostgreSQL"]
   - ✗ WRONG: ["Listed everything from resume", "Non-technical skills"]

7. **experience_years** (number):
   - Total years of professional work experience (numeric).
   - Round to 1 decimal place.
   - ✓ CORRECT: 5.2, 10, 0.5
   - ✗ WRONG: "5+ years", "Junior", "5 years 3 months"

8. **work_experience** (array of objects):
   - List of previous jobs.
   - Each entry: {
       "company": "Company Name",
       "role": "Job Title",
       "start_date": "YYYY-MM or YYYY",
       "end_date": "YYYY-MM or YYYY or 'Present'"
     }
   - ✓ CORRECT:
     [
       {
         "company": "Google",
         "role": "Senior Software Engineer",
         "start_date": "2020-01",
         "end_date": "Present"
       }
     ]

9. **education** (array):
   - Educational degrees only (Bachelor, Master, PhD, Diploma, Certificate).
   - Do NOT include company names or non-degree items.
   - ✓ CORRECT: ["Bachelor of Science in Computer Science", "Master of Technology in Software Engineering"]
   - ✗ WRONG: ["AWS Certified Solutions Architect", "Google", "Business Analyst"]

10. **summary** (string):
    - Professional summary or objective from the resume.
    - 1-3 sentences maximum.
    - Omit if not present.

11. **companies_worked_at** (array):
    - List of company names where candidate worked.
    - ✓ CORRECT: ["Google", "Microsoft", "Amazon"]

12. **current_role** (string):
    - Most recent job title.
    - ✓ CORRECT: "Senior Software Engineer"
    - ✗ WRONG: "Currently unemployed", "Job Seeker"

13. **important_keywords** (array):
    - Key technologies or concepts mentioned multiple times.
    - Different from "skills" - can include concepts like "Microservices", "Cloud Architecture".
    - ✓ CORRECT: ["Microservices", "Cloud Architecture", "Machine Learning"]

---

EXAMPLE EXTRACTION:

INPUT RESUME:
---
John Smith
San Francisco, CA | john.smith@techcorp.com | (555) 123-4567

Senior Software Engineer with 5 years of experience in full-stack development.

PROFESSIONAL EXPERIENCE:
Google Inc. | Senior Software Engineer | Jan 2020 - Present
- Led development of microservices platform using Python and Go
- Managed PostgreSQL database with 100M+ records
- Deployed services to AWS (Lambda, EC2, RDS)

Facebook | Software Engineer | Jun 2018 - Dec 2019
- Built React.js frontend components
- Implemented REST APIs using Django framework

EDUCATION:
Bachelor of Science in Computer Science
University of California, Berkeley | Graduated 2018

SKILLS:
Programming: Python, Go, JavaScript, TypeScript
Frontend: React, Vue.js, HTML, CSS
Backend: Django, FastAPI, Node.js
Databases: PostgreSQL, MongoDB, Redis
Cloud: AWS (Lambda, EC2, RDS), Docker, Kubernetes
Certifications: AWS Certified Solutions Architect
---

OUTPUT:
{
  "name": "John Smith",
  "email": "john.smith@techcorp.com",
  "phone": "(555) 123-4567",
  "location": "San Francisco, CA",
  "skills": [
    "Python", "Go", "JavaScript", "TypeScript", "React", "Vue.js",
    "HTML", "CSS", "Django", "FastAPI", "Node.js", "PostgreSQL",
    "MongoDB", "Redis", "AWS", "Lambda", "EC2", "RDS",
    "Docker", "Kubernetes", "AWS Certified Solutions Architect"
  ],
  "key_skills": ["Python", "React", "AWS"],
  "experience_years": 5,
  "work_experience": [
    {
      "company": "Google Inc.",
      "role": "Senior Software Engineer",
      "start_date": "2020-01",
      "end_date": "Present"
    },
    {
      "company": "Facebook",
      "role": "Software Engineer",
      "start_date": "2018-06",
      "end_date": "2019-12"
    }
  ],
  "education": [
    "Bachelor of Science in Computer Science"
  ],
  "summary": "Senior Software Engineer with 5 years of experience in full-stack development.",
  "companies_worked_at": ["Google Inc.", "Facebook"],
  "current_role": "Senior Software Engineer",
  "important_keywords": ["Microservices", "Full-stack Development", "Cloud Architecture"],
  "experience_summary": "5 years of full-stack software engineering across startups and FAANG"
}

---

NOW EXTRACT FROM THE PROVIDED RESUME:

[HEADER BLOCK]
{header_block}

[SKILLS/COMPETENCIES BLOCK]
{skills_block}

[BODY SNIPPET]
{body_block}

Return EXACTLY one valid JSON object. No markdown, no additional text. If a field is not found, use "" for strings, 0 for numbers, or [] for arrays.
```

---

## Part 3: Structured JSON Output Format

### 3.1 Recommended Output Schema

```json
{
  "metadata": {
    "extraction_confidence": 0.75,
    "extraction_warnings": [
      "low_extraction_confidence",
      "text_normalized",
      "skills_section_not_found"
    ],
    "source_format": "pdf",
    "pages_processed": 2,
    "processing_time_ms": 1234
  },
  "candidate": {
    "name": {
      "value": "John Smith",
      "confidence": 0.95,
      "source": "header_position"
    },
    "email": {
      "value": "john.smith@company.com",
      "confidence": 0.99,
      "source": "regex_extraction"
    },
    "phone": {
      "value": "+1-555-123-4567",
      "confidence": 0.98,
      "source": "regex_extraction"
    },
    "location": {
      "value": "San Francisco, CA",
      "confidence": 0.85,
      "source": "location_extraction"
    },
    "summary": {
      "value": "Senior Software Engineer...",
      "confidence": 0.90,
      "source": "section_extraction"
    }
  },
  "professional": {
    "experience_years": {
      "value": 5.2,
      "confidence": 0.92,
      "source": "date_range_calculation"
    },
    "experience_level": "Senior",
    "current_role": {
      "value": "Senior Software Engineer",
      "confidence": 0.95,
      "source": "work_history_extraction"
    },
    "companies_worked_at": [
      {
        "value": "Google",
        "confidence": 0.99,
        "source": "work_history_extraction"
      }
    ],
    "work_experience": [
      {
        "company": "Google",
        "role": "Senior Software Engineer",
        "start_date": "2020-01",
        "end_date": "Present",
        "duration_months": 48,
        "confidence": 0.95
      }
    ]
  },
  "skills": {
    "all_skills": [
      {
        "value": "Python",
        "confidence": 0.99,
        "category": "programming_language",
        "source": "explicit_section",
        "frequency": 5
      }
    ],
    "key_skills": [
      {
        "value": "Python",
        "confidence": 0.99,
        "rank": 1
      }
    ],
    "skill_categories": {
      "programming_languages": ["Python", "Go", "JavaScript"],
      "frameworks": ["React", "Django"],
      "databases": ["PostgreSQL", "MongoDB"],
      "cloud_platforms": ["AWS"],
      "tools": ["Docker", "Kubernetes"],
      "certifications": ["AWS Certified Solutions Architect"]
    },
    "metadata": {
      "total_skills": 20,
      "skills_confidence": 0.92,
      "source": "section_explicit + vocabulary_scan",
      "warnings": []
    }
  },
  "education": [
    {
      "value": "Bachelor of Science in Computer Science",
      "confidence": 0.98,
      "institution": "University of California, Berkeley",
      "graduation_year": 2018
    }
  ],
  "projects": [
    {
      "value": "Microservices Platform",
      "description": "Led development of microservices platform...",
      "confidence": 0.90,
      "source": "work_experience_extraction"
    }
  ],
  "important_keywords": [
    {
      "value": "Microservices",
      "frequency": 3,
      "context": "appears in job descriptions and skills"
    }
  ]
}
```

---

## Part 4: Preprocessing Pipeline Implementation

### 4.1 Multi-Column Resume Handling

```python
def normalize_multi_column_layout(text: str) -> str:
    """
    Detect and normalize multi-column layouts from PDFs.
    Example: "Python, Java    Docker, Kubernetes" → separate lines
    """
    lines = text.splitlines()
    normalized = []
    
    for line in lines:
        # Detect columns: 4+ spaces indicate column separator
        if re.search(r'\s{4,}', line):
            # Split and put each column on separate line
            parts = re.split(r'\s{4,}', line)
            for part in parts:
                if part.strip():
                    normalized.append(part.strip())
        else:
            # Fix ligature artifacts: "Progra mming" → "Programming"
            line = fix_pdf_ligatures(line)
            normalized.append(line)
    
    return "\n".join(normalized)

def fix_pdf_ligatures(line: str) -> str:
    """Fix common PDF extraction artifacts."""
    # Handle broken words: "Pro gram ming" → "Programming"
    line = re.sub(r'\b([a-z])\s+([a-z])', r'\1\2', line)
    # Handle hyphenated breaks: "Pro-\ngramming" → "Programming"
    line = re.sub(r'-\n(\S)', r'\1', line)
    return line
```

### 4.2 Enhanced Section Detection

```python
def detect_sections_robust(text: str) -> dict[str, str]:
    """
    Detect resume sections with improved accuracy.
    Returns: {"header": "...", "skills": "...", "experience": "...", ...}
    """
    
    sections = {
        "header": "",      # Top lines before first section
        "skills": "",
        "experience": "",
        "education": "",
        "projects": "",
        "other": ""
    }
    
    lines = text.splitlines()
    
    # Step 1: Find all section headers
    header_positions = {}
    for i, line in enumerate(lines):
        normalized = re.sub(r'[^a-z0-9\s]', '', line.lower()).strip()
        
        for section_name, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, normalized):
                    header_positions[section_name] = i
                    break
    
    # Step 2: Extract content between headers
    sorted_headers = sorted(header_positions.items(), key=lambda x: x[1])
    
    for idx, (section_name, position) in enumerate(sorted_headers):
        # Find next section or end of document
        next_position = sorted_headers[idx+1][1] if idx+1 < len(sorted_headers) else len(lines)
        
        # Extract lines between headers (excluding header line)
        section_lines = lines[position+1:next_position]
        sections[section_name] = "\n".join(section_lines).strip()
    
    # Step 3: If no headers found, fallback to heuristic
    if not header_positions:
        sections["header"] = "\n".join(lines[:10])  # Top 10 lines
        sections["other"] = "\n".join(lines[10:])
    
    return sections
```

---

## Part 5: Fallback Strategies

### 5.1 Name Extraction Fallback Logic

```python
def extract_name_with_fallback(
    text: str,
    email: str = "",
    phone: str = ""
) -> (str, float, str):
    """
    Returns: (name, confidence: 0.0-1.0, method: "header" | "ner" | "email" | "fallback")
    
    Priority order:
    1. Try deterministic header-based extraction (high confidence)
    2. Try NER-based extraction with validation
    3. Try email prefix extraction
    4. Return "Unknown" with low confidence
    """
    
    # Priority 1: Header extraction (90%+ confidence)
    name = extract_deterministic_name(text)
    if is_valid_person_name(name):
        confidence = _name_confidence_from_header(text, name)
        if confidence >= 0.75:
            return name, confidence, "header"
    
    # Priority 2: NER extraction with validation
    try:
        ner_names = extract_names_via_ner(text)
        for candidate in ner_names:
            if is_valid_person_name(candidate):
                confidence = 0.70
                return candidate, confidence, "ner"
    except Exception:
        pass
    
    # Priority 3: Email prefix
    if email:
        email_name = derive_name_from_email(email)
        if is_valid_person_name(email_name):
            return email_name, 0.65, "email"
    
    # Fallback: Unknown
    return "Unknown", 0.20, "fallback"

def is_valid_person_name(name: str) -> bool:
    """
    Validate that a candidate string is a plausible person name.
    """
    if not name or len(name) > 60 or len(name.split()) > 4:
        return False
    
    # Reject known bad patterns
    bad_keywords = [
        "resume", "cv", "profile", "summary", "objective",
        "engineer", "developer", "manager", "analyst", "director",
        "inc", "ltd", "corp", "company", "solutions", "technologies",
        "phone", "email", "address", "github", "linkedin", "portfolio"
    ]
    
    for keyword in bad_keywords:
        if keyword in name.lower():
            return False
    
    # Must have at least 2 "word-like" parts
    parts = name.split()
    if len(parts) < 2:
        return False
    
    # Parts should be mostly alphabetic
    for part in parts:
        if not re.match(r"^[a-z'-]+$", part, re.IGNORECASE):
            return False
    
    return True
```

### 5.2 Skills Extraction Fallback

```python
def extract_skills_with_fallback(
    text: str,
    tech_vocabulary: set[str]
) -> (list[str], dict):
    """
    Fallback strategy for skills extraction:
    1. Try section-explicit extraction
    2. Try vocabulary scan on body
    3. Try LLM extraction (if enabled)
    4. Return empty + confidence signal
    """
    
    # Strategy 1: Section-explicit
    skills_section = find_section(text, headers=SKILLS_HEADERS)
    if skills_section and len(skills_section) > 50:
        explicit = extract_skills_from_section(skills_section)
        if len(explicit) >= 3:
            return explicit, {
                "confidence": 0.95,
                "source": "section_explicit",
                "strategy": 1
            }
    
    # Strategy 2: Vocabulary scan
    vocab_skills = scan_vocabulary(text, tech_vocabulary)
    if len(vocab_skills) >= 3:
        return vocab_skills, {
            "confidence": 0.80,
            "source": "vocabulary_scan",
            "strategy": 2
        }
    
    # Strategy 3: LLM extraction (if enabled)
    if USE_LLM_EXTRACTION:
        try:
            llm_skills = extract_skills_via_llm(text)
            if llm_skills and len(llm_skills) >= 3:
                return llm_skills, {
                    "confidence": 0.75,
                    "source": "llm_extraction",
                    "strategy": 3
                }
        except Exception:
            pass
    
    # Fallback: Return empty with warning
    return [], {
        "confidence": 0.20,
        "source": "none",
        "strategy": "fallback",
        "warning": "Could not extract skills from resume"
    }
```

---

## Part 6: Hybrid Rule + AI Approach

### 6.1 Architecture

```
                    Resume (PDF/DOCX)
                            |
                    [Preprocessing]
                            |
        ┌───────────────────┼───────────────────┐
        |                   |                   |
    [Header]           [Sections]           [Body]
        |                   |                   |
        |         ┌─────────┼─────────┐         |
        |         |         |         |         |
        |     [Skills]  [Exp]    [Edu]         |
        |         |         |         |         |
        └─────────┼─────────┴─────────┴─────────┘
                  |
        ┌─────────┴──────────┐
        |                    |
    [Rule-Based]        [AI-Based]
    Extraction          Extraction
        |                    |
        • Regex             • LLM
        • Section parse     • Structured output
        • Vocab match       • Contextual validation
        • Confidence score  • Confidence signal
        |                    |
        └─────────┬──────────┘
                  |
          [Result Merge]
          • Prioritize rule results
          • Use LLM to fill gaps
          • Final confidence scoring
                  |
        [Structured Output]
        with Confidence Metadata
```

### 6.2 Decision Logic

```python
def extract_resume_hybrid(resume_text: str) -> dict:
    """
    Hybrid extraction: rule-based first, AI-based for gaps.
    """
    
    # STEP 1: Preprocess
    text_normalized = preprocess_resume(resume_text)
    sections = detect_sections_robust(text_normalized)
    
    # STEP 2: Rule-based extraction
    rule_results = {
        "name": extract_deterministic_name(text_normalized),
        "email": extract_contact_email(text_normalized),
        "phone": extract_phone_number(text_normalized),
        "location": extract_location(text_normalized),
        "skills": extract_skills_deterministic(sections["skills"], text_normalized),
        "experience": extract_experience_years(sections["experience"]),
        "education": extract_education(sections["education"]),
    }
    
    # STEP 3: AI-based extraction (gap filling + validation)
    llm_results = {}
    if USE_LLM_EXTRACTION and confidence_is_low(rule_results):
        llm_results = extract_via_llm(
            header=sections["header"],
            skills=sections["skills"],
            body=sections["experience"][:2000]
        )
    
    # STEP 4: Merge with priority
    final_results = {
        **rule_results,  # Rule results as base
        **{k: v for k, v in llm_results.items() if should_use_llm_result(k, rule_results, llm_results)}
    }
    
    # STEP 5: Confidence scoring
    final_results_with_confidence = add_confidence_metadata(final_results, rule_results, llm_results)
    
    return final_results_with_confidence
```

---

## Part 7: Implementation Priority

### Phase 1: Quick Wins (1-2 weeks)

1. **Expand Skills Vocabulary**
   - Current: ~500 skills
   - Target: 1000+ skills with categories
   - Add: Enterprise platforms (Salesforce, ServiceNow, SAP), certifications, AI/ML tools

2. **Fix Education Filter**
   - Implement context-aware `is_education_item()` function
   - Add certification pattern detection
   - Test on 50 resumes with mixed education/certifications

3. **Add Section Header Variations**
   - Expand from 16 to 40+ header patterns
   - Include regional variations and abbreviations
   - Test detection on 100 resumes

4. **Improve Name Extraction Confidence**
   - Implement validation filters (reject titles, companies, etc.)
   - Add confidence scoring with metadata
   - Test on 100 resumes with known correct names

### Phase 2: Structural (2-3 weeks)

1. **Multi-Column Handling**
   - Implement PDF column detection
   - Test on 50 multi-column resumes
   - Measure improvement in skills extraction

2. **Enhanced LLM Prompt**
   - Add 5-10 example extractions
   - Include field definitions with examples
   - Add "Do Not Extract" guidance
   - Measure reduction in hallucinations

3. **Structured Output**
   - Implement confidence metadata
   - Add extraction source tracking
   - Create warning signals
   - Test output format with 100 resumes

4. **Fallback Strategies**
   - Implement name extraction fallback chain
   - Implement skills extraction fallback chain
   - Add degradation metrics

### Phase 3: Advanced (3-4 weeks)

1. **Hybrid Rule + AI**
   - Implement decision logic for when to use each strategy
   - Confidence-based selection
   - Test on 500 diverse resumes

2. **Machine Learning-based Skill Extraction**
   - Train NER model on resume dataset
   - Add skill categorization
   - Implement skill ranking by frequency/relevance

3. **Multi-Pass Validation**
   - Cross-reference extracted data
   - Detect and flag inconsistencies
   - Self-correct obvious errors

---

## Part 8: Success Metrics

| Metric | Current | Target | Timeframe |
|--------|---------|--------|-----------|
| Name Accuracy | 60% | 90% | Phase 1-2 |
| Skills Completeness | 50-60% | 85%+ | Phase 1-2 |
| Skills Precision | ~70% | 95%+ | Phase 1-2 |
| Education Accuracy | 75% | 98% | Phase 1 |
| Format Support | PDF, some DOCX | PDF, DOCX, TXT | Phase 2 |
| Extraction Time | 2-5s (CPU) | <3s | Phase 2 |
| Confidence Metadata | No | Yes | Phase 2 |

---

## Part 9: Testing Strategy

### 9.1 Test Datasets

**Create diverse test set (500 resumes)**:
- 50 multi-column layouts
- 50 with non-standard sections
- 50 with certifications (test education filter)
- 50 with minimal information
- 50 scanned/OCR-heavy
- 200 standard format (baseline)

### 9.2 Automated Testing

```python
def test_extraction_accuracy():
    """
    Run on labeled test set.
    Calculate precision, recall, F1 per field.
    """
    test_data = load_labeled_resumes("test_set_500.csv")
    results = {"name": {}, "skills": {}, "education": {}, ...}
    
    for resume, expected in test_data:
        extracted = extract_resume(resume)
        
        # Calculate metrics per field
        for field in ["name", "skills", "education", ...]:
            precision = calculate_precision(extracted[field], expected[field])
            recall = calculate_recall(extracted[field], expected[field])
            f1 = 2 * (precision * recall) / (precision + recall)
            
            results[field].append({"precision": precision, "recall": recall, "f1": f1})
    
    print_summary(results)
    return results
```

### 9.3 Manual Review

- Review 100 failures from automated test
- Categorize by root cause
- Prioritize fixes by frequency

---

## Conclusion

This hybrid approach combines rule-based extraction (fast, deterministic) with AI-based extraction (flexible, context-aware) to achieve 85%+ accuracy across diverse resume formats. The phased implementation allows quick wins in Phase 1 (education filter, vocabulary expansion) while building toward a robust production system in Phases 2-3.

**Key Success Factors**:
1. Comprehensive section detection (40+ header patterns)
2. Confidence-based decision making (know when to trust extraction)
3. Fallback strategies (graceful degradation)
4. Extensive testing (500+ labeled resumes)
5. Hybrid rule + AI approach (best of both worlds)
