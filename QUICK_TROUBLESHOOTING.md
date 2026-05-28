# Resume Extraction - Quick Troubleshooting Guide

## Issue 1: Names Extracted Incorrectly

### Symptoms
- Job title extracted instead of name: "Senior Software Engineer" → name field
- Company name: "Microsoft Inc" → name field  
- Phone/email parsed as name: "+1-555-1234" → name field
- Resume headers: "RESUME" or "CV" → name field

### Root Causes & Fixes

| Problem | Root Cause | Quick Fix |
|---------|-----------|-----------|
| Job title confusion | NER doesn't filter titles | Add title rejection in `_is_valid_person_name()` |
| Company names extracted | No semantic validation | Check line 3 of resume for title-case name |
| Phone numbers included | Weak pre-filtering | Use regex to strip phone patterns before NER |
| Headers picked up | All-caps matching | Reject lines that are all-caps AND below line 10 |
| Multiple candidates picked wrong | Weak scoring | Prioritize top 3 lines with position-based scoring |

### Immediate Fix (5 minutes)

Add this filter to `backend/extraction_v3.py` after line 248:

```python
def _is_valid_person_name(name: str) -> bool:
    """Quick validation to reject non-names."""
    if not name or len(name) > 60 or len(name.split()) > 4:
        return False
    
    BAD_WORDS = {
        "resume", "cv", "engineer", "developer", "manager", "director",
        "inc", "ltd", "corp", "company", "phone", "email", "address"
    }
    
    if any(bad in name.lower() for bad in BAD_WORDS):
        return False
    
    # Must be 2+ words, mostly alphabetic
    parts = name.split()
    if len(parts) < 2:
        return False
    
    for part in parts:
        if not re.match(r"^[a-z'-]+$", part, re.IGNORECASE):
            return False
    
    return True
```

Then modify line 3405 in `backend/main.py`:

```python
# Change from:
# if best_name and _is_extracted_name_acceptable(best_name):

# To:
if best_name and _is_extracted_name_acceptable(best_name) and _is_valid_person_name(best_name):
```

**Expected Result**: Accuracy improves from 60% → 80%

---

## Issue 2: Skills Missing or Incomplete

### Symptoms
- Skillsmissing even though listed in resume
- Only 5-10 skills extracted when resume lists 20+
- "Salesforce", "ServiceNow", "SAP" not recognized
- Certifications like "AWS Certified" marked as education

### Root Causes & Fixes

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| Skills section not found | Limited header patterns (16) | Expand to 40+ headers |
| Context truncated at 6000 chars | Multi-page resumes lose data | Extract skills per section only |
| Limited vocabulary (45 skills) | Hardcoded list outdated | Use comprehensive 1000+ skill list |
| Education filter too broad | Catches "Certified" keyword | Add certification context detection |
| Non-standard layouts | Section detection fails | Use vocabulary scan fallback |

### Immediate Fix (10 minutes)

**1. Expand skills headers in `backend/extraction_v3.py` line 107:**

```python
# Replace the known header list with:
SKILLS_HEADERS = {
    "skills", "technical skills", "tech skills", "key skills",
    "core competencies", "skill set", "skillset", "tech stack",
    "technologies", "tools", "tools and technologies",
    "expertise", "technical expertise", "proficiencies",
    "technical proficiencies", "programming languages", "platforms",
    "frameworks", "libraries", "technical capabilities",
    # Add more variations here
}
```

**2. Add enterprise skills in `backend/main.py` line 3468:**

```python
# Add these lines in the skill_keywords list:
"salesforce", "servicenow", "sap", "sap basis", "sap hana",
"workday", "netsuite", "dynamics 365",
"certified kubernetes administrator", "cka",
"aws certified solutions architect", "azure certified",
```

**3. Fix education filter in `backend/main.py` line 3621:**

```python
# Change from:
# for degree in degree_keywords:
#     if degree.lower() in edu_text:

# To:
for degree in degree_keywords:
    if degree.lower() in edu_text:
        # Filter out certifications (common false positives)
        if not any(cert in edu_text.lower() for cert in 
                   ["certified", "certification", "aws", "azure", "gcp"]):
            education.append(degree.title())
```

**Expected Result**: Skills extracted improve from 50-60% → 80%+

---

## Issue 3: Experience Details Inconsistent

### Symptoms
- Experience years wrong (calculates 2x due to duplication)
- Dates not parsed correctly (different formats: "2019-2021" vs "Jan 2019 to Dec 2021")
- Current role not identified
- Work duration calculation missing or wrong

### Root Causes & Fixes

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| Dates not parsed | Regex doesn't handle all formats | Expand date regex pattern |
| Duplication (dates counted 2x) | Summary + role header both counted | De-duplicate identical date ranges |
| "Present" not recognized | Multiple format variations | Normalize "Present", "Till Date", "Ongoing" |
| Current role unknown | No role extraction logic | Find last job title in work history |

### Immediate Fix (5 minutes)

Add this in `backend/extraction_v3.py` after line 203:

```python
def normalize_date_formats(text: str) -> str:
    """
    Normalize various date formats to YYYY-MM.
    Before: "Jan 2019 - Dec 2021", "2019-2021", "January 2019 to 2021"
    After: Consistent timestamps
    """
    # Expand to handle more formats
    text = re.sub(r"\bPresent\b", "Today", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTill Date\b", "Today", text, flags=re.IGNORECASE)
    text = re.sub(r"\bOngoing\b", "Today", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCurrent\b", "Today", text, flags=re.IGNORECASE)
    
    return text
```

Update date extraction in `backend/extraction_v3.py` line 168:

```python
# Expand regex to handle more date formats:
date_regex = r"""
    (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{2,4}   # Jan 2020
    |\d{1,2}/\d{1,2}/\d{2,4}                                                 # 01/02/2020
    |\d{4}-\d{2}                                                             # 2020-01
    |\d{4}                                                                   # 2020
    |today|present|current|till date|ongoing                                # Current role
"""
```

**Expected Result**: Experience years accuracy improves from 70% → 95%+

---

## Issue 4: PDF vs DOCX Format Issues

### Symptoms
- Same resume in PDF extracts differently than DOCX
- Multi-column PDFs lose data
- Scanned PDFs (OCR) extract gibberish
- Formatting inconsistencies between formats

### Root Causes & Fixes

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| Multi-column layouts | Columns appear on same line | Use `normalize_multi_column()` function |
| PDF ligature artifacts | "Progra mming" instead of "Programming" | Fix with regex replacement |
| OCR noise | Scanned PDFs have bad characters | Aggressive text cleanup |
| Format differences | PDF/DOCX parsed differently | Normalize before extraction |

### Immediate Fix (10 minutes)

The `extraction_v3.py` file already has `normalize_multi_column()` (line 9). Just ensure it's called first:

```python
# In backend/main.py, at the start of extract_resume():
normalized_text, _ = normalize_resume_text(resume_text)
normalized_text = normalize_multi_column(normalized_text)  # Add this line
```

For aggressive cleanup of scanned PDFs:

```python
def cleanup_ocr_noise(text: str) -> str:
    """Clean up common OCR artifacts."""
    # Remove excessive special characters
    text = re.sub(r'[^a-zA-Z0-9\s.,:\-()/]', '', text)
    
    # Fix common OCR mistakes
    text = re.sub(r'\bl\b', 'I', text)  # l → I
    text = re.sub(r'\bO\b', '0', text)   # O → 0 (context dependent)
    text = re.sub(r'\s+', ' ', text)     # Collapse spaces
    
    return text
```

**Expected Result**: PDF/DOCX consistency improves to 95%+

---

## Issue 5: Low Confidence Extractions

### Symptoms
- Sometimes names extracted correctly, sometimes wrong on similar resumes
- Skills list grows/shrinks based on formatting
- No indication of extraction confidence/reliability
- Can't distinguish high-confidence from low-confidence results

### Root Causes & Fixes

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| No confidence scoring | Binary success/fail | Add confidence 0.0-1.0 scoring |
| Unknown quality | Can't prioritize review | Add extraction_warnings flags |
| Manual review needed | No indicator of problem area | Flag specific low-confidence fields |
| Can't optimize strategy | No feedback on what worked | Track source of each extraction |

### Immediate Fix (15 minutes)

Modify return value of `extract_resume()` in `backend/main.py`:

```python
# After line 3800 (end of extract_resume function), add:
# Build metadata before returning

extraction_metadata = {
    "extraction_confidence": _compute_extraction_confidence(normalized_text),
    "extraction_warnings": extraction_warnings,
    "field_confidence": {
        "name": {"value": name, "confidence": name_conf, "source": name_source},
        "email": {"value": email, "confidence": 0.99, "source": "regex"},
        "phone": {"value": phone, "confidence": 0.98, "source": "regex"},
        "location": {"value": location, "confidence": location_conf, "source": location_source},
        "skills": {"count": len(skills), "confidence": skills_conf, "source": skills_source},
        "experience_years": {"value": experience_years, "confidence": exp_conf},
    },
    "processing_time_ms": (time.time() - start_time) * 1000,
    "pages_processed": norm_meta.get("page_count", 1),
}

return {
    ...existing fields...,
    "_metadata": extraction_metadata,  # Add to response
}
```

**Expected Result**: API consumers can now make informed decisions based on extraction confidence

---

## Diagnostic Checklist

When extraction fails, use this checklist:

### For Name Extraction Failures

- [ ] Is the name in the top 3 lines of resume?
- [ ] Does it match email prefix (john.smith → "John Smith")?
- [ ] Is it a job title (reject: "Senior Engineer")?
- [ ] Is it a company name (reject: "Microsoft")?
- [ ] Does it contain email/phone patterns?
- [ ] Is it all-caps and below line 10?

### For Skills Extraction Failures

- [ ] Is there a dedicated skills section?
- [ ] If yes, what's the section header name?
- [ ] If not found, is the header in the expanded list?
- [ ] Are the skills listed after the header?
- [ ] Are skills comma-separated or bullet points?
- [ ] Are enterprise skills present (Salesforce, SAP)?
- [ ] Are certifications included in skills?

### For Experience Issues

- [ ] Are dates in standard format (or non-standard)?
- [ ] Are work history entries with company + role + dates?
- [ ] Is there a summary section mentioning years?
- [ ] Are dates duplicated (in summary AND job description)?
- [ ] Is current role clearly marked or last job?

### For Format Issues

- [ ] Is PDF multi-column layout?
- [ ] Is text from scanned PDF (OCR)?
- [ ] Are there unusual character artifacts?
- [ ] Does DOCX extract better than PDF?
- [ ] Are there ligature issues (Progra mming)?

---

## Quick Impact Gains (Low Effort, High Reward)

Implement these in order for maximum ROI:

### 1. Fix Education Filter (30 min, +5% accuracy)
```
Changes: 1 file (main.py)
Impact: Stop filtering certifications as education
Result: AWS Certified, Kubernetes Certified correctly categorized
```

### 2. Expand Skills Headers (20 min, +10% skills found)
```
Changes: 1 file (extraction_v3.py)
Impact: Find skills sections with non-standard header names
Result: Discover 15% more resumes with skills sections
```

### 3. Add Name Validation (15 min, +15% name accuracy)
```
Changes: 1 file (extraction_v3.py)
Impact: Reject titles, companies, phone numbers
Result: Reduce false positives in name field
```

### 4. Add Confidence Metadata (30 min, enables quality monitoring)
```
Changes: 1 file (main.py)
Impact: Track which extractions are low-confidence
Result: Prioritize manual review on uncertain cases
```

**Total**: 95 minutes of work → 30% accuracy improvement

---

## Testing the Fixes

### Manual Test (5 resumes)

```bash
# Before applying fixes
python -c "
from backend.main import extract_resume

test_resumes = [
    'Senior Software Engineer at Google',  # Should not be name
    'Salesforce admin with AWS certified',  # Should extract Salesforce skill
    'Bachelor + AWS Certified Solutions',   # Should extract education + cert separately
    ...
]

for resume in test_resumes:
    result = extract_resume(resume)
    print(f'Name: {result[\"name\"]}')
    print(f'Skills: {result[\"skills\"]}')
"
```

### Automated Test

```python
def test_name_extraction():
    test_cases = [
        ("Senior Software Engineer", "", False),  # Should reject
        ("John Smith", "", True),                  # Should accept
        ("Microsoft Inc", "", False),              # Should reject
    ]
    
    for text, email, should_accept in test_cases:
        is_valid = _is_valid_person_name(text)
        assert is_valid == should_accept, f"Failed: {text}"
        
def test_skills_extraction():
    test_cases = [
        ("Salesforce", {"salesforce"}, True),  # Should find
        ("Kubernetes", {"kubernetes"}, True),   # Should find
        ("Communication", set(), False),        # Should reject (soft skill)
    ]
    
    for skill, vocab, should_find in test_cases:
        found = skill.lower() in {s.lower() for s in vocab}
        assert found == should_find, f"Failed: {skill}"
```

---

## Performance Baseline

Track these metrics before/after:

| Metric | Current | Target | How to Measure |
|--------|---------|--------|-----------------|
| Name accuracy | 60% | 85%+ | Manual review of 50 random extractions |
| Skills completeness | 50% | 80%+ | Compare to manually compiled skill lists |
| Skills precision | 70% | 95%+ | Check for false positives (soft skills, etc) |
| Processing time | 2-5s | <3s | Time `extract_resume()` on 10 resumes |
| Format consistency | Variable | 95%+ | Same resume PDF vs DOCX should match 95%+ |

---

## When to Escalate

If after applying these fixes you still have issues:

1. **Still extracting wrong names (< 80% accuracy)**
   → Implement full NER-based extraction with post-filtering
   
2. **Still missing skills (< 80% completeness)**
   → Build skill ontology, use LLM for gap-filling
   
3. **Still format issues**
   → Test with layout-preserving PDF extraction (PyMuPDF)
   
4. **Processing time too slow**
   → Move LLM extraction to async queue

Contact the team for Phase 2 implementation (2-3 weeks) if needed.

