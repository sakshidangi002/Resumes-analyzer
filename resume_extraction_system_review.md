## 1. Current Pipeline Explanation

This section reconstructs the actual end‑to‑end extraction flow based on the current code (`backend/main.py`, `backend/api.py`, `frontend/app.py`) and your described stack.

### 1.1 High‑level flow

- **Resume upload (Streamlit → FastAPI)**
  - `frontend/app.py` sends uploaded PDF/DOCX files to `POST /upload` (`backend/api.py`).
- **Text extraction**
  - `upload_resume()` writes each uploaded file to a temp path and calls:
    - `extract_text_from_pdf(path)` (pdfplumber) or
    - `extract_text_from_docx(path)` (python‑docx),
  - Then deletes the temp file.
- **Resume text → structured extraction (LLM + heuristics)**
  - For each valid text, `upload_resume()` calls `extract_resume(resume_text)` in `backend/main.py`.
  - `extract_resume()` performs:
    1. **Regex baselines / heuristics**
       - Email and phone regexes.
       - Name extraction via:
         - `extract_name_candidates()` + `select_best_name()` (candidate scoring, optional spaCy NER).
         - Fallback to `extract_name_from_header()` if candidate‑based extraction throws.
       - Location extraction via `extract_location_from_text()`.
       - Skill keyword scan from a large `skill_keywords` list.
       - Education keywords and very simple projects extraction.
    2. **LLM extraction (NuExtract)**
       - Builds context using `build_extraction_context()`:
         - `header_block`: first 35 non‑empty lines.
         - `skills_block`: text under the first skills header, from `_find_section_block()` using `_SKILLS_SECTION_HEADERS`.
         - `body_block`: first 6000 characters of the resume.
       - Crafts a detailed prompt with:
         - **Strict JSON schema** (name, email, phone, location, skills, education, experience fields, etc.).
         - Explicit rules about:
           - Name must not be headings (RESUME, CV, job titles) and must exclude phone/email.
           - Technical skills vs soft skills vs education.
           - `key_skills` vs `skills` distinction.
       - Uses `AutoTokenizer` + `AutoModelForCausalLM` (`numind/NuExtract-1.5-smol`) to generate JSON.
       - Parses the JSON by scanning for the first balanced `{ ... }` block, then `json.loads`.
       - If the JSON is not a template/placeholder, fields from `llm_result` can overwrite:
         - `name`, `email`, `phone`, `location`, `skills`, `education`, `projects`,
           `experience_summary`, `experience_years`, `work_experience`, `summary`,
           `companies_worked_at`, `current_role`, `important_keywords`, `key_skills`.
       - The **LLM name** is accepted only if `_is_plausible_person_name()` says it looks like a person.
    3. **Experience estimation fallbacks**
       - If `experience_summary` or `experience_years` are missing or 0:
         - Regex for “X years” in free text.
         - `_estimate_years_from_raw_dates()`:
           - Extracts an experience section (`_extract_experience_section()`) using headers like `experience`, `work experience`, etc.
           - Scans for date ranges (`Jan 2020 – Feb 2024`, `2018–2022`, etc.).
           - Uses `calculate_experience_years()` to merge overlapping ranges and compute total years.
    4. **Skills validation and normalization**
       - Re‑parses explicit skills section via `extract_skills_from_text(resume_text)`:
         - Uses `_SKILLS_SECTION_HEADERS` to find the first skills section.
         - Collects up to 20 lines under this header, splitting by bullets and separators,
           filtering soft skills and noise.
       - Merges `section_skills` + `skills` from keyword scan/LLM.
       - Applies several filters:
         - `_is_education_like_phrase()` to drop degree/university items.
         - `_is_percentage_metric()` to drop “React 80%”, “Management 75%”.
         - `canonicalise_skill()` for `.NET`, `ASP.NET`, etc.
         - `_skill_is_supported_by_text()`:
           - Keeps only skills that actually appear in the resume text (using normalized word‑boundary matching).
       - Repeats the same pipeline for `key_skills`.
       - Applies `_boost_domain_skills()` to move `.NET` or `React` higher in the lists if roles/project descriptions emphasize them.
       - Classifies into:
         - `primary_skills` vs `other_skills` via `classify_skills(skills, key_skills)`.
    5. **Name fallback from email**
       - If `name` is still empty or `"Unknown"`, derives a title‑cased name from the email local part (e.g. `john.doe` → `John Doe`).
  - Returns a rich dict:
    - `name`, `email`, `phone`, `location`,
    - `skills`, `primary_skills`, `other_skills`, `key_skills`,
    - `experience_years`, `experience_summary`,
    - `education`, `projects`, `summary`,
    - `companies_worked_at`, `current_role`, `important_keywords`,
    - `extraction_warnings` (e.g. `name_low_confidence`).
- **Enrichment**
  - In `upload_resume()`, you currently call `_enrichment_fallback(extracted)` (not the LLM enrichment) to derive:
    - `summary`, `one_liner`, `experience_line`, `experience_tags` from the extracted fields.
- **Persistence (PostgreSQL + ChromaDB)**
  - Builds `ResumeSchema` from the extracted data (Pydantic validation).
  - Performs duplicate checks by `email` or `phone`.
  - Writes the original file to `backend/uploads/` and sets `resume_link`.
  - Builds an `embedding_text` (name, contact, location, skills, experience, summary, education, role, companies, keywords).
  - Uses `SentenceTransformer("all-MiniLM-L6-v2")` to generate an embedding.
  - Stores embedding and metadata to Chroma (`PersistentClient(path="./chromadb")`).
  - Inserts a row into `resumes` table with all extracted fields plus enrichment.
- **Frontend consumption**
  - Streamlit reads `/resumes`, `/resume/{id}`, `/resumes/compare`, `/chat`, `/resume/{id}/chat`.
  - Tabs include:
    - Upload, Resumes table, Compare, Global Chat, Resume‑scoped Chat, Export.
  - UI allows re‑extracting skills and experience for existing resumes via:
    - `/resumes/{resume_id}/reextract-skills`
    - `/resumes/{resume_id}/reextract-experience`
    - `/resumes/{resume_id}/reextract-key-skills`

---

## 2. Root Causes of Extraction Failures

This section links your observed issues to specific behaviors in the current implementation.

### 2.1 Name extraction failures

Symptoms:
- Name at the very top is ignored.
- Headings like “Resume”, “Software Engineer”, “Profile” sometimes appear as the name.
- Lines where name and phone/email appear together are mis‑handled.

Relevant logic:
- `extract_name_candidates(resume_text, email)`:
  - Collects candidates from:
    - spaCy NER (`PERSON` entities) if `en_core_web_sm` is installed.
    - Top 25 header lines + lines around first email/phone (top 40 lines).
    - Email local part.
  - For header lines it:
    - Strips contact noise via `_strip_contact_noise()` (emails, phones, URLs).
    - Splits on separators `| • / – -` and adds each part as candidate.
- `select_best_name(candidates, resume_text, email)`:
  - Scores candidates with positive signals:
    - Appears near the top.
    - 2–4 words, title case.
    - Overlap with email local part.
  - Negative signals:
    - Tokens in `_HEADER_BAD_NAMES` (e.g. `resume`, `curriculum vitae`, `profile`, job titles).
    - Institution words (`university`, `college`, etc.).
    - Role words (`developer`, `engineer`, `manager`, etc.).
    - Degree words (`bachelor`, `masters`, etc.).
    - All‑caps headings, digits, “X of Y” phrases.
  - Rejects candidates with final score `< 0.2` or that map to `_HEADER_BAD_NAMES`.
  - If score is low, falls back to a name derived from the email.
- Fallback: `extract_name_from_header(resume_text)` scans only the **top 25 non‑empty lines**, chooses the first line that `_is_plausible_person_name()` accepts, after stripping contact noise and separators.
- LLM can overwrite name if it returns something that `_is_plausible_person_name()` accepts.

**Root causes:**

- **(R1) Single‑word or unusual names are systematically rejected.**
  - `_is_plausible_person_name()` requires **2–4 words**:
    - A resume whose first line is `JOHN` or `PRIYA` (single word) will never pass.
    - If spaCy is absent or fails, and the email local part is not a good proxy, the system ends up with `"Unknown Candidate"` or a wrong fallback.

- **(R2) Over‑strict header heuristics when layout is noisy.**
  - `_looks_like_header()` treats any short all‑caps line (3–40 chars, no digits) as a section header.
  - Many PDF layouts emit the top name line as all caps (`JOHN DOE`), which then:
    - Gets treated as a header and sometimes rejected as a name (via `_looks_like_header()` check).
    - Or competes with other header‑like lines (“PROFILE”, “SUMMARY”) in candidate scoring.
  - In some PDFs, “RESUME” or job titles appear in similar positions, and because the scoring is only **relative**, if the real name is lost/merged with other tokens during text extraction, the best remaining candidate can be a title‑like line.

- **(R3) Fragility to PDF text ordering and line merging.**
  - `pdfplumber`’s `page.extract_text()` often:
    - Merges multi‑column layouts incorrectly.
    - Places contact details on the same line as the name (e.g. `John Doe | +91 ... | john@x.com`).
    - Sometimes emits “RESUME” as the first non‑empty line, followed by contact info, and only later the name (e.g. inside a positioned text box).
  - While `_strip_contact_noise()` tries to clean emails/phones, your heuristics still assume that:
    - Top 25 logical lines roughly follow human reading order.
    - The name appears before major headings.
  - When that assumption fails, the candidate set becomes polluted with many non‑name lines, and scoring may pick a header or job title.

- **(R4) Optional spaCy dependency leads to mode drift.**
  - If `en_core_web_sm` is not installed, `_get_nlp_ner()` logs a warning and disables NER:
    - `extract_name_candidates()` then relies exclusively on header lines + email local part.
    - On machines where spaCy is present vs absent, behavior can be **very different**, explaining inconsistent results across environments, even with the same resume.

- **(R5) LLM name override is still structure‑dependent.**
  - The LLM sees only:
    - `header_block` (first 35 lines),
    - `skills_block`,
    - `body_block` (first 6000 chars).
  - If the name is not clearly visible in `header_block` (e.g. misordered PDF text or heavy formatting), the LLM might:
    - Infer a name from email or hallucinate.
    - Or misinterpret a prominent title as the name.
  - Even though `_is_plausible_person_name()` is applied, it cannot catch all cases (e.g. exotic job titles that look like multi‑word names).

### 2.2 Skills extraction failures

Symptoms:
- Skills section clearly exists but `skills` is empty.
- Education or project text appears in `skills`.
- Bullet‑formatted skills are ignored.

Relevant logic:
- **Skill section detection:** `extract_skills_from_text(resume_text)`:
  - Scans all lines, finds the first where `_norm_header(line)` is in `_SKILLS_SECTION_HEADERS`.
    - `_SKILLS_SECTION_HEADERS` includes many variations:
      - `"skills"`, `"technical skills"`, `"tech stack"`, `"core competencies"`, `"technical proficiencies"`, etc.
  - Collects up to 20 lines after that header, until:
    - Two consecutive blank lines, or
    - Another `_looks_like_header()` is encountered.
- **Skill tokenization:**
  - Strips leading bullet characters.
  - If there is a colon, uses the right‑hand side (to skip labels like `Technical Skills:`).
  - Splits by common separators: commas, pipes, semicolons, bullets.
  - Filters out:
    - Very short (`len <= 1`) or very long (`len > 60`) tokens.
    - Pure digits.
  - Drops tokens that match `_SOFT_SKILL_STOP` (communication, teamwork, etc.).
  - Preserves acronyms and brand‑case names where possible.
- **Validation:** later, in `extract_resume()`:
  - Merges `section_skills` with keyword/LLM skills.
  - Uses `_is_education_like_phrase()`, `_is_percentage_metric()`, and `_skill_is_supported_by_text()` to filter.

**Root causes:**

- **(R6) Section detection requires a clean, explicit skills header.**
  - `extract_skills_from_text()` only works if one of the lines matches `_SKILLS_SECTION_HEADERS` exactly after normalization.
  - Resumes with:
    - Non‑standard headings (“Technical Snapshot”, “Professional Technologies”, “Summary of Expertise”).
    - Skills embedded under other sections (“Key Projects and Skills” or only inside `Experience` bullets).
  - …will **never trigger** `extract_skills_from_text()`, so you fall back to:
    - Keyword scan over a predefined list.
    - LLM skills (which are later heavily filtered).

- **(R7) Heavy validation can zero‑out skills when layout is noisy or wording is non‑canonical.**
  - `_skill_is_supported_by_text()` normalizes both text and skill to alphanumeric tokens and requiring:
    - Full word match for short tokens (`java`, `sql`), or
    - All words present for multi‑word skills (`react native`, `asp.net mvc`).
  - This is good to avoid hallucinations, but:
    - If the text extraction merges words or breaks them with line breaks / hyphens (`Reac\n t` or `ASP . NET`), the normalized skill may **not** match.
    - If the LLM returns abstract skills not literally present (e.g. `Web Development`, `Backend Engineering`), they are filtered out.
  - Combined with `_is_education_like_phrase()` and `_is_percentage_metric()`, you may drop many borderline tokens, resulting in **empty skill lists** even though the resume clearly lists technologies.

- **(R8) Only the **first** skills section is used.**
  - If a resume has multiple skill‑like sections:
    - `Technical Skills`,
    - Later a separate `Tools & Technologies` table,
  - `_find_section_block()` and `extract_skills_from_text()` always stop at the *first* matching header.
  - If that first section is short or nearly empty, further skills might be ignored.

- **(R9) Skills embedded purely in experience/project text rely on keyword lists and LLM.**
  - Many resumes list skills only inside bullets under experience:
    - “Led development of a microservices platform using Go, Kubernetes, and gRPC…”
  - Your current pipeline:
    - Does not have a dedicated detector for “skills in body text”.
    - Depends on:
      - Keyword scan across `skill_keywords` (hard‑coded list).
      - LLM skills, which are then filtered by `_skill_is_supported_by_text()`.
  - Any technology not in `skill_keywords` but present in text is only reachable via LLM and can easily be filtered out during validation.

### 2.3 Resume structure variability

Symptoms:
- Different labels for skills: “Skills”, “Technical Skills”, “Tech Stack”, “Core Competencies”, etc.
- Skills sometimes exist only inside experience sections, or in paragraphs, or as inline text.
- Section boundaries are not consistently identified.

Relevant logic:
- Section detection is essentially:
  - `_find_section_block(resume_text, headers)`:
    - Linear search for a line whose normalized header is in a **fixed set**.
    - Collects lines until:
      - Double blank, or
      - Another `_looks_like_header()` is seen.
  - `_looks_like_header()`:
    - Matches `_COMMON_SECTION_HEADERS` or `_SKILLS_SECTION_HEADERS`.
    - Or uses the “short all‑caps” heuristic.
- Experience section isolation uses `_extract_experience_section()`, but education, projects, etc. do not have equivalent robust boundary isolation.

**Root causes:**

- **(R10) Section detection is entirely lexicon‑based with limited fuzzy/semantic support.**
  - You have broad coverage for common headers, but:
    - Any resume that uses novel headings, creative wording, or mixed languages will not be recognized.
    - Headers that include extra qualifiers (“Technical Skills & Interests”, “Key Technical Competencies”) may not normalize exactly into the known sets.
  - When section detection fails:
    - LLM sees skills only as scattered tokens in the `body_block`, without a clearly demarcated skills region.
    - Rule‑based skill extraction (`extract_skills_from_text`) returns empty.

- **(R11) No structural model of the document – everything is flat lines.**
  - Because `extract_text_from_pdf()` uses plain `page.extract_text()`:
    - No concept of **columns**, **tables**, **indentation**, or **font size/weight** is preserved.
  - Section detection relies entirely on text content and capitalization; any layout‑only cue (e.g. big bold heading in a different font) is lost.
  - Multi‑column resumes are especially problematic: lines may interleave content from different logical sections.

### 2.4 Inconsistent extraction quality across resumes

Symptoms:
- Some resumes parse flawlessly; others fail on name/skills or produce minimal information.

**Root causes (combined):**

- **(R12) Heterogeneous dependencies and environment differences.**
  - spaCy NER is optional; its presence or absence changes name candidate sets.
  - Chroma/embedding and LLM models are loaded lazily; if there are timeouts or partial failures, fallbacks behave differently.

- **(R13) LLM behavior is strongly conditioned on partial, layout‑dependent context.**
  - `header_block` and `skills_block` are derived from text extraction heuristics.
  - When these heuristics are correct, the LLM has a clean view and performs well.
  - When they are wrong (bad ordering, missed skills header), the LLM works with noisy or incomplete context, and:
    - Sometimes still infers reasonable output.
    - Sometimes fails completely or returns generic placeholders (which you filter).

- **(R14) Aggressive post‑processing sacrifices recall for precision.**
  - Validation steps are tuned to reduce hallucination and misclassification (e.g. not counting education as skills), which is good.
  - But in ambiguous/rescue cases where only the LLM can infer structure, these filters can:
    - Throw away weak but correct signals.
    - Leave `skills` empty or minimal.

---

## 3. Code‑Level Weaknesses

This section highlights specific functions/modules where the above issues originate and what each currently does.

### 3.1 `extract_text_from_pdf` / `extract_text_from_docx` (backend/main.py)

- **What it does**
  - `extract_text_from_pdf(path)`: loops over pages with pdfplumber, concatenates `page.extract_text()` with `\n`.
  - `extract_text_from_docx(path)`: joins non‑empty paragraph texts with `\n`.
- **Why it fails**
  - PDF text extraction is naive:
    - Ignores document layout (columns, tables, headers/footers).
    - Uses default pdfplumber settings (no custom `layout=True` or coordinate‑aware reconstruction).
  - Many resumes rely on columns (left column contact details / skills, right column experience).
  - The resulting line order can be **very different** from human reading order, which breaks:
    - Name/section detection (top 25 lines might be scrambled).
    - Skills detection (skills may interleave with unrelated text).
- **What needs improvement**
  - Introduce a more layout‑aware PDF strategy, for example:
    - Use `extract_words()` with x/y coordinates and cluster into reading order.
    - Use specialized libraries or commercial parsers designed for resumes.
    - At least add heuristics for:
      - Detecting columns (by x‑coordinate gaps).
      - Keeping header region (top ~15% of page) as a unit for name/contact extraction.

### 3.2 Name extraction helpers (backend/main.py)

Key functions: `_strip_contact_noise`, `_is_plausible_person_name`, `extract_name_from_header`, `extract_name_candidates`, `select_best_name`.

- **What they do**
  - Clean lines of emails/phones/URLs before evaluating them as name candidates.
  - Use regular expressions and dictionaries of bad names, role words, institution/degree words.
  - Score candidates by:
    - Shape (2–4 words, title case, no digits).
    - Position (top 10–15 lines).
    - Overlap with email local part.
- **Why they fail**
  - Over‑constrained notion of a “valid” name:
    - Single‑word names and some non‑Western naming conventions are rejected.
    - All‑caps names (common in resumes) are penalized or dropped.
  - Scores are relative: if all candidates are bad due to layout issues, the “least bad” candidate can still be a heading or job title.
  - There is no explicit modeling of typical resume header patterns like:
    - First non‑blank **non‑heading** line above contact line is likely the name.
    - Lines that contain “resume/cv/profile” should be demoted even if they superficially look like names.
  - SpaCy dependency is optional, leading to different behavior across environments.
- **What needs improvement**
  - Relax constraints:
    - Allow 1–4 tokens; treat single‑word names as low‑confidence but valid candidates.
    - Adjust caps handling: for top lines, an all‑caps line surrounded by non‑caps lines is often the name.
  - Add structural patterns:
    - Look at the line directly **above** the first email/phone as a strong name candidate.
    - Use proximity to contact details more heavily than generic header heuristics.
  - Make spaCy optional but not critical:
    - Use NER primarily as a **tie‑breaker**, not the core discovery mechanism.

### 3.3 Skills extraction helpers (backend/main.py)

Key functions: `_SKILLS_SECTION_HEADERS`, `_looks_like_header`, `_find_section_block`, `extract_skills_from_text`, `_is_education_like_phrase`, `_is_percentage_metric`, `_skill_is_supported_by_text`, `canonicalise_skill`, `classify_skills`, `_boost_domain_skills`.

- **What they do**
  - Detect one skills‑like section and parse it into a cleaned list of skills.
  - Merge skills from:
    - Hard‑coded dictionary scan,
    - LLM output,
    - Skills section.
  - Filter skills against resume text and remove education/metrics noise.
  - Classify top 3 primary skills and others.
- **Why they fail**
  - Heavy reliance on explicit, exact skills headers.
  - Only the first skills section is used.
  - No fallback to “skills scattered through experience” beyond:
    - Hard‑coded `skill_keywords`.
    - LLM outputs that are then filtered.
  - Hard filters, while valuable for precision, make recall fragile when:
    - Text normalization breaks matching.
    - LLM outputs slightly abstract but valid skills.
- **What needs improvement**
  - Extend section detection:
    - Fuzzy header matching with similarity scores (e.g. Levenshtein distance).
    - Secondary pass that looks for dense sequences of technology tokens even without a named header.
  - Add a dedicated **skills‑from‑body** extractor:
    - Identify technology tokens from experience/project sentences using a tech lexicon + simple NER tags (ORG/PRODUCT).
    - Use co‑occurrence and context cues (e.g. verbs like “implemented using X and Y”).
  - Make validation thresholds configurable and allow “low‑confidence” skills to be surfaced separately instead of dropped.

### 3.4 Section detection (`_find_section_block`, `_extract_experience_section`) (backend/main.py)

- **What they do**
  - `_find_section_block()`:
    - Finds a line whose normalized form matches a header from a fixed set.
    - Returns up to `max_lines` lines after the header, stopping at blank streaks or next header.
  - `_extract_experience_section()`:
    - Specialized for experience: similar header search, up to next header.
- **Why they fail**
  - For both skills and experience, this approach:
    - Assumes a well‑behaved sequence of headers in a single column layout.
    - Has no notion of nested sections or repeated headings.
  - Many resumes pack:
    - Skills inside `Professional Summary` or `Experience` sections.
    - Multiple mini blocks with tech stacks under project descriptions.
  - Without any semantic density detection, these are invisible as “sections”.
- **What needs improvement**
  - Layered section detection:
    - First pass: rule‑based header detection (what you have).
    - Second pass: **density‑based** detection of tech tokens (lines with many tech terms are likely skills/tool descriptions).
    - Third pass: LLM‑assisted section labeling on pre‑segmented blocks (small, cheap models).

### 3.5 LLM prompt design and context (backend/main.py → `extract_resume`)

- **What it does**
  - Provides a verbose prompt with explicit rules and schema.
  - Passes:
    - `HEADER BLOCK`, `SKILLS/COMPETENCIES BLOCK`, `BODY SNIPPET`.
  - Asks NuExtract to return the JSON structure with all fields.
- **Why it fails**
  - Context is **heavily dependent** on earlier preprocessing:
    - If `skills_block` is empty or misleading, the LLM has no explicit skills region.
    - If `header_block` is polluted by poor PDF extraction, the LLM sees a noisy top region.
  - Prompt is static; it does not adapt to:
    - Missing sections.
    - Very long resumes where important info lies beyond the first 6000 characters.
  - Name rules are present, but the model may still violate them, requiring post‑hoc rejection.
- **What needs improvement**
  - Make the LLM stage **section‑aware**:
    - First detect logical blocks and pass them as labeled snippets: `header`, `contacts`, `skills`, `experience`, `projects`, `education`.
    - Condition prompts based on which sections are actually detected (e.g. “There is no explicit skills section; infer skills from experience paragraphs.”).
  - Use:
    - A **smaller extraction model** for repeated local tasks (per section).
    - A single global pass only for complex summarization, not for all fields at once.

### 3.6 Post‑processing & validation (backend/main.py → `extract_resume`)

- **What it does**
  - Protects against:
    - LLM hallucination.
    - Misclassified education as skills.
    - Percentage/proficiency scales as skills.
  - Ensures final data is normalized and deduplicated.
- **Why it fails**
  - There is no explicit **confidence scoring** or layering:
    - You either accept or drop skills/names/sections.
    - A user cannot see “model thought this was a possible skill but not supported by text”.
  - All validation is hard‑threshold based and not tuned per field; e.g. name extraction is binary (either plausible name or fallback).
- **What needs improvement**
  - Add a **confidence‑aware validation layer**:
    - E.g. mark skills as `high_confidence` if found in explicit skills section, `medium` if repeated in experience, `low` if only inferred by LLM.
    - Persist `extraction_warnings` and confidence for name, skills, and experience.
  - Allow frontend to surface these signals (e.g. “Name inferred from email; please verify”).

---

## 4. Resume Parsing Challenges

To make the architecture robust, we need to explicitly handle these inherent challenges:

- **Highly variable layouts**
  - Single column vs multi‑column.
  - Tables, text boxes, sidebars.
  - Headers and footers repeating on every page.
- **Inconsistent section labeling**
  - Skills presented as:
    - Dedicated section headers with bullet lists.
    - Sub‑sections under experience or summary.
    - Embedded in prose (“experienced in X, Y, Z”).
  - Custom headings and mixed languages.
- **Text extraction noise**
  - Out‑of‑order text from PDFs.
  - Broken words, ligatures, hyphenation.
  - Missing punctuation or bullet markers.
- **Ambiguous tokens**
  - Names that look like organizations and vice versa.
  - Skills that are also common words (e.g. “Excel”, “Word”).
  - Degrees vs certifications.
- **LLM behavior**
  - Tendency to “fill in the blanks” unless strongly constrained.
  - Sensitivity to prompt structure and context length.

Your current system addresses some of these via heuristics and validation, but:
- It lacks a **formal representation of sections**.
- It relies on **a single global LLM call** rather than multiple smaller, context‑aware passes.
- It does not use **semantic similarity** or **embeddings** to normalize and detect skills beyond a static lexicon.

---

## 5. Recommended Architecture (Hybrid System)

Below is a more reliable architecture tailored for high‑variance resume formats while reusing much of your existing code.

### 5.1 Proposed pipeline

**1. Document ingestion & layout‑aware parsing**
- For PDFs:
  - Use a parser that provides token coordinates (pdfplumber `extract_words`, PyMuPDF, or a dedicated resume parser).
  - Reconstruct **reading order** per page and detect columns:
    - Cluster words by x‑coordinate gaps into left/right columns.
    - Within each column, sort by y.
  - Preserve:
    - Page number,
    - Column index,
    - Line index.
- For DOCX:
  - Keep paragraph styles where possible (e.g. heading vs body).

**Output:** list of `blocks`, each with:
- `text`, `page`, `column`, `line_index`, `style` (optional).

**2. Text cleaning & normalization**
- Normalize whitespace, remove control chars.
- Handle hyphenation (`“micro‑\nservices” → “microservices”`).
- Strip long repeated headers/footers by pattern.
- Keep both:
  - `plain_text` (for global regex and LLM).
  - `blocks` (for structural heuristics).

**3. Section detection (hybrid rules + semantics)**
- Rule‑based:
  - Use a **much broader** dictionary of possible headings, but with fuzzy matching:
    - Edit distance <= 2 from known labels.
    - Case insensitive.
  - Consider:
    - Block position (top of document → header/contact).
    - Font/size/weight (if available).
    - All‑caps and surrounded by blank lines as candidate headings.
- Semantic:
  - For ambiguous headings:
    - Use sentence‑embeddings (MiniLM, etc.) to compare to prototype phrases for `skills`, `experience`, `education`, `summary`.
    - Assign each block a section label with confidence.
- Detect:
  - `header/contact`
  - `summary/profile`
  - `skills/tools`
  - `experience`
  - `projects`
  - `education`
  - `certifications`

**Output:** structured document model:
- `sections = {label: [blocks...]}`, with ordering and confidence.

**4. Rule‑based extraction per field**

**Name & contact (rule‑first):**
- Use header/contact section(s) plus first page top region:
  - Analyze lines around the first email/phone detection:
    - Candidate = line above/below contact line that is not a known heading.
  - Apply relaxed rules:
    - 1–4 tokens, mostly alphabetic.
    - Not matching “resume/cv/profile/software engineer” via a fuzzy title lexicon.
  - Use spaCy NER (when available) as an auxiliary signal, not primary.

**Skills (rule‑first):**
- From `skills/tools` sections:
  - Parse bullets and comma‑separated lists similar to `extract_skills_from_text`, but:
    - Support multiple skills sections.
    - Allow nested subsections (e.g. `Programming Languages: ...`, `Tools: ...`).
- From `experience`, `projects`, `summary`:
  - Use:
    - Tech lexicon (extended, domain‑specific).
    - Regex for patterns like `using X`, `with Y`, `in Z`.
  - Build a **skills evidence map**:
    - For each skill, record:
      - Sections where it appears,
      - Number of occurrences,
      - Surrounding verbs (implemented/built/maintained).

**Education, experience, projects:**
- Use section labels and date patterns to extract:
  - Degree names and institutions.
  - Job entries (company, role, dates).
  - Project titles and stacks.

**Output:** preliminary structured object with fields similar to your current schema but **without any LLM involvement yet**.

**5. LLM enhancer (section‑aware, not monolithic)**

Instead of one huge prompt, use **targeted prompts**:

- **Name & header refinement**
  - Only if rule‑based name is low confidence (e.g. single token, no email overlap).
  - Prompt: “Given this header/contact block, what is the candidate’s full name?” with strict constraints.
- **Skills expansion & grouping**
  - Provide:
    - List of detected skills with evidence (where found, counts).
  - Ask model to:
    - Group into categories.
    - Suggest 3–5 `key_skills` consistent with evidence, not inventing new ones.
- **Experience summarization**
  - For each extracted job list, ask model to:
    - Summarize experience.
    - Compute approximate years if dates are ambiguous.

Critically:
- **LLM never invents raw facts** that do not have evidence from earlier stages:
  - Use your existing `_skill_is_supported_by_text()` or an improved evidence‑based filter.

**6. Validation & normalization layer**

- Combine rule‑based and LLM outputs with explicit **confidence scores**:
  - For each field, keep:
    - `value`,
    - `source` (`rule`, `llm`, `fallback_email`, etc.),
    - `confidence` (`high`, `medium`, `low`).
- Apply normalization:
  - Skill canonicalization (`.Net` → `.NET`, `ReactJS` → `React`).
  - Name canonicalization (trim spaces, remove embedded contact info).
  - Deduplication, ordering.
- Persist warnings:
  - `name_confidence: low`, `skills_from_body_only: true`, `skills_section_missing: true`, etc.
  - Store them in DB so UI can surface them.

**7. Final structured output**

Exactly as your `extract_resume` returns today, but with:
- Additional metadata:
  - Confidence per field.
  - Source attribution.
  - Detected sections summary.

---

## 6. Implementation Strategy (Step‑By‑Step)

This section outlines how to evolve your existing code towards the architecture above without a massive refactor, while targeting your concrete pain points.

### 6.1 Improve name extraction

1. **Relax ` _is_plausible_person_name` conditions**
   - Allow 1–4 tokens.
   - Treat all‑caps 2–3 token lines near the top as valid candidates (with a penalty rather than hard rejection).
2. **Strengthen positional heuristics**
   - Within `extract_name_candidates()`:
     - Mark the line immediately above and/or below the first email/phone match as high‑priority candidates.
     - Demote lines that:
       - Contain typical section words (`resume`, `profile`, `summary`) even if they pass shape checks.
3. **Normalize environment behavior**
   - On startup, check for spaCy; if not available:
     - Set a flag and use purely rule‑based name extraction **consistently** (no half‑on/half‑off behavior).
4. **Add optional LLM name refinement**
   - Only when:
     - The rule‑based name has `low_confidence`.
   - Use a tiny, constrained prompt on header block only, and ensure:
     - Result is accepted only if it still passes a stricter name validator.

### 6.2 Improve skills extraction

1. **Extend section detection**
   - Expand `_SKILLS_SECTION_HEADERS` with additional patterns, but more importantly:
   - Add fuzzy matching:
     - Normalize line to lowercase, strip non‑alphanumerics.
     - Compute similarity with known labels (Levenshtein/Jaro‑Winkler).
     - Accept as a skills header if similarity > threshold (e.g. 0.8).
2. **Support multiple skills sections**
   - Refactor `extract_skills_from_text` to:
     - Find **all** blocks whose headers are skills‑like.
     - Aggregate skills from all these blocks.
3. **Introduce “skills from body” module**
   - New helper that, given experience and project sections, uses:
     - Extended tech lexicon.
     - Simple patterns (`using X`, `with Y`, `(X, Y, Z)` after tech lead verbs).
   - Merge these skills with section‑based skills but track their source (for confidence).
4. **Tweak validation thresholds**
   - For skills found in explicit skills sections:
     - Trust them more; relax `_skill_is_supported_by_text()` or base it on proximity rather than full doc search.
   - For skills only in body (experience/projects):
     - Keep current strict validation.
5. **Add skill confidence tags**
   - For each skill, store:
     - `source_section: skills | experience | projects | summary | llm_only`.
     - `confidence: high | medium | low`.

### 6.3 Enhance section detection

1. **Generalize `_find_section_block`**
   - Accept a list of label prototypes and allow:
     - Fuzzy/semantic matching of headings.
   - Add:
     - `section_label` and `confidence`.
2. **Add density‑based detection**
   - Look for blocks of 3+ consecutive lines where:
     - The ratio of known tech tokens to total tokens is high.
   - Tag such blocks as implicit skills sections even without explicit “Skills” headings.
3. **Persist section metadata**
   - Return a simple `sections` map from the preprocessor:
     - This can feed both rule‑based extraction and LLM prompts.

### 6.4 Safer LLM usage

1. **Decompose global prompt**
   - Split `extract_resume` into:
     - Rule‑only core extraction.
     - Optional LLM passes:
       - Name refinement.
       - Skills grouping/normalization.
       - Experience summarization.
2. **Use evidence‑only prompts**
   - For skills:
     - Provide only the candidate skills and their evidence snippets.
     - Ask LLM to classify/group, not to invent new ones.
3. **Add systematic error handling for LLM JSON**
   - Already present partially (template detection).
   - Extend with:
     - `schema` validation (e.g. `pydantic` model) before accepting fields.
     - Fallback to rule‑only output when parsing fails.

### 6.5 UI & validation integration

1. **Expose extraction warnings and confidence**
   - Extend DB model to store:
     - `name_confidence`, `skills_confidence`, `sections_detected`, etc.
   - Show in UI:
     - Icons/badges for low‑confidence fields.
2. **Provide one‑click re‑parse with updated pipeline**
   - You already have `reextract_*` endpoints.
   - After pipeline upgrade, these can:
     - Re‑run the new logic on older resumes.
     - Record any differences in a change log if desired.

---

## 7. Advanced Improvements for Production‑Level Parsing

Finally, ways to make the system robust across arbitrary formats and future‑proof.

### 7.1 Semantic understanding + keyword detection + structural heuristics

- **Semantic understanding**
  - Use embeddings to:
    - Classify blocks as `skills`, `experience`, `education`, etc. even when headings are missing or creative.
    - Group similar skills (e.g. `JS`, `JavaScript`, `Node`).
- **Keyword detection**
  - Maintain and periodically update a **skills ontology**:
    - Programming languages, frameworks, tools, platforms, certifications.
    - Map each to canonical forms and synonyms.
  - Use this ontology for:
    - Rule‑based detection.
    - Semantically grouping skills for UI/analytics.
- **Structural heuristics**
  - Combine:
    - Position (top vs body vs bottom of page).
    - Spacing and blank lines.
    - Bullet density.
  - E.g. lines with:
    - High bullet density and many tech tokens → likely skills.
    - Dates and companies → experience.

### 7.2 Test harness & regression suite

- Build an internal corpus of **diverse resumes**:
  - Multiple languages, layouts, PDFs generated from Word vs design tools.
  - Edge cases: single‑page vs multi‑page, no explicit skills section, only paragraphs.
- For each resume, maintain:
  - Ground truth for name, email, phone, core skills, and years of experience.
- After each pipeline change:
  - Run automated evaluation:
    - Precision/recall for skills.
    - Accuracy for name and contact fields.
    - Error analysis reports (e.g. which resumes lost skills after changes).

### 7.3 Model & data versioning

- Version the extraction pipeline:
  - `extraction_version` column in DB.
  - Store:
    - Which rules version, skills ontology version, and model version were used.
- Allow re‑running new versions on stored files:
  - Using existing `/reextract-*` endpoints or a batch job.

### 7.4 Performance & reliability considerations

- Keep heavy LLM calls off the request path where possible:
  - Use background workers (Celery/RQ) for enrichment and heavy extraction.
  - Frontend polls for status or uses notifications.
- Cache:
  - Embeddings for identical documents (e.g. re‑uploads).
  - LLM responses for identical `(resume_id, operation_type)` combinations.

---

## 8. Summary: Why Extraction Fails Today & How the Architecture Fixes It

- **Name extraction fails** because:
  - Heuristics are overly strict and sensitive to layout noise.
  - SpaCy dependency is optional and changes behavior.
  - LLM fallback is fed with poorly segmented context.
- **Skills extraction fails** because:
  - Section detection is lexicon‑only and only uses the first found section.
  - Skills embedded in experience/summary are not first‑class citizens.
  - Aggressive validation can remove correct but weakly evidenced skills.
- **Section detection fails** because:
  - It assumes simple single‑column, clearly labeled resumes.
  - It ignores semantic similarity and tech density.

The proposed hybrid architecture:
- Introduces a **layout‑aware parser** and **explicit section model**.
- Uses **rule‑based extraction** as the foundation, with LLMs only as enhancers constrained by evidence.
- Adds **confidence, validation, and metadata** instead of binary accept/reject, making behavior consistent and debuggable.
- Provides clear extension points (skills ontology, semantic headers, regression tests) so you can reliably handle new resume formats over time.

