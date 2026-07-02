import re
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import Counter

logger = __import__('logging').getLogger(__name__)

# ============================================================================
# SKILL FILTERING RULES
# ============================================================================

# Sections to NEVER extract skills from
EXCLUDED_SECTIONS = {
    "education", "educational", "academic", "qualification", "qualifications",
    "languages", "language", "language proficiency",
    "interests", "hobbies", "personal interests", "activities",
    "personal details", "contact", "contact information", "address",
    "certifications", "certificates", "achievements", "awards",
    "references", "referee", "personal information"
}

# Human languages to reject
HUMAN_LANGUAGES = {
    "english", "hindi", "punjabi", "french", "german", "spanish", "italian",
    "portuguese", "russian", "chinese", "japanese", "korean", "arabic",
    "bengali", "telugu", "marathi", "tamil", "gujarati", "kannada",
    "malayalam", "urdu", "odia", "assamese", "nepali", "sinhala",
    "thai", "vietnamese", "indonesian", "malay", "filipino", "swahili",
    "dutch", "danish", "swedish", "norwegian", "finnish", "polish",
    "czech", "hungarian", "romanian", "bulgarian", "greek", "turkish",
    "hebrew", "persian", "pashto", "dari"
}

# Hobbies to reject
HOBBIES = {
    "reading", "cooking", "cricket", "music", "traveling", "travelling",
    "photography", "dancing", "singing", "painting", "drawing", "gardening",
    "fishing", "hiking", "camping", "swimming", "cycling", "running",
    "gaming", "video games", "chess", "cards", "puzzles", "yoga",
    "meditation", "knitting", "sewing", "crafting", "writing", "blogging",
    "watching movies", "watching tv", "listening to music", "socializing",
    "volunteering", "mentoring", "teaching", "learning", "collecting"
}

# University/College keywords to reject
UNIVERSITY_KEYWORDS = {
    "university", "college", "institute", "institutes", "school", "schools",
    "academy", "academies", "polytechnic", "institute of technology",
    "university of", "college of", "school of"
}

# Location keywords to reject
LOCATION_KEYWORDS = {
    "india", "punjab", "delhi", "mumbai", "bengaluru", "bangalore",
    "hyderabad", "pune", "chennai", "kolkata", "chandigarh", "mohali",
    "shimla", "dharamshala", "himachal", "himachal pradesh",
    "maharashtra", "karnataka", "tamil nadu", "telangana", "gujarat",
    "rajasthan", "uttar pradesh", "bihar", "west bengal", "madhya pradesh",
    "kerala", "andhra pradesh", "odisha", "jammu", "kashmir", "goa",
    "usa", "uk", "uae", "canada", "australia", "germany", "france",
    "singapore", "japan", "china", "south korea", "new zealand"
}

# Generic words to reject
GENERIC_WORDS = {
    "the", "and", "or", "with", "using", "for", "from", "at", "in", "on",
    "by", "of", "to", "a", "an", "as", "be", "is", "are", "was", "were",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "this", "that", "these", "those",
    "best", "good", "better", "well", "very", "much", "many", "more", "most",
    "some", "any", "all", "each", "every", "both", "either", "neither",
    "make", "made", "make", "making", "used", "use", "using", "project",
    "projects", "work", "working", "worked", "team", "teams", "group",
    "groups", "company", "companies", "organization", "organizations",
    "business", "businesses", "client", "clients", "customer", "customers",
    "service", "services", "product", "products", "solution", "solutions",
    "system", "systems", "application", "applications", "software", "softwares",
    "development", "developing", "developed", "management", "managing",
    "managed", "operations", "operational", "operation", "support", "supporting"
}

# Technical skill categories (whitelist)
TECHNICAL_CATEGORIES = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "perl", "r", "matlab",
    "lua", "haskell", "elixir", "erlang", "clojure", "f#", "dart", "julia",
    
    # Web Frameworks
    "django", "flask", "fastapi", "spring", "spring boot", "express", "react",
    "angular", "vue", "vue.js", "svelte", "next.js", "nuxt.js", "gatsby",
    "ember", "backbone", "jquery", "asp.net", "laravel", "symfony", "rails",
    
    # Frontend
    "html", "css", "html5", "css3", "sass", "scss", "less", "tailwind", "bootstrap",
    "material ui", "ant design", "chakra ui", "bulma", "foundation",
    
    # Backend
    "node.js", "nodejs", "deno", "bun", "nest.js", "nestjs", "express.js",
    
    # Databases
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "neo4j", "influxdb", "timescaledb", "cockroachdb",
    "mariadb", "oracle", "sql server", "mssql", "firestore", "supabase",
    
    # Cloud Platforms
    "aws", "amazon web services", "azure", "gcp", "google cloud", "heroku",
    "digitalocean", "linode", "vultr", "ibm cloud", "alibaba cloud", "oracle cloud",
    
    # DevOps
    "docker", "kubernetes", "k8s", "jenkins", "gitlab ci", "github actions",
    "circleci", "travis ci", "terraform", "ansible", "chef", "puppet",
    "nagios", "prometheus", "grafana", "elk", "elk stack", "nginx", "apache",
    
    # AI/ML
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "jupyter", "spacy", "nltk", "opencv", "pillow",
    "langchain", "langgraph", "openai", "claude", "crewai", "rag", "hugging face",
    "transformers", "bert", "gpt", "llm", "machine learning", "deep learning",
    "nlp", "computer vision", "reinforcement learning", "mlops", "mlflow",
    
    # Data Engineering
    "airflow", "dbt", "spark", "hadoop", "kafka", "hive", "presto", "trino",
    "snowflake", "bigquery", "redshift", "databricks", "fivetran", "airbyte",
    
    # Testing
    "pytest", "jest", "mocha", "jasmine", "selenium", "cypress", "playwright",
    "junit", "testng", "robot framework", "appium",
    
    # Version Control
    "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
    
    # Mobile
    "react native", "flutter", "android", "ios", "swift", "kotlin", "dart",
    "xamarin", "ionic", "cordova",
    
    # Message Queues
    "rabbitmq", "kafka", "activemq", "redis streams", "aws sqs", "aws sns",
    
    # API
    "rest", "restful", "graphql", "grpc", "soap", "openapi", "swagger",
    
    # Authentication
    "oauth", "jwt", "saml", "ldap", "openid connect",
    
    # Monitoring
    "datadog", "new relic", "splunk", "sentry", "rollbar", "pagerduty",
    
    # Package Managers
    "npm", "yarn", "pnpm", "pip", "conda", "poetry", "composer", "maven", "gradle",
    
    # IDEs
    "vscode", "visual studio", "intellij", "pycharm", "eclipse", "xcode",
    
    # Operating Systems
    "linux", "ubuntu", "debian", "centos", "rhel", "fedora", "windows", "macos",
    "unix", "bsd", "android", "ios"
}


def is_excluded_section(section_name: str) -> bool:
    """Check if a section should be excluded from skill extraction."""
    if not section_name:
        return False
    section_lower = section_name.lower().strip()
    return any(excluded in section_lower for excluded in EXCLUDED_SECTIONS)


def is_human_language(skill: str) -> bool:
    """Check if a skill is a human language."""
    skill_lower = skill.lower().strip()
    return skill_lower in HUMAN_LANGUAGES


def is_hobby(skill: str) -> bool:
    """Check if a skill is a hobby."""
    skill_lower = skill.lower().strip()
    return skill_lower in HOBBIES


def is_university_or_college(skill: str) -> bool:
    """Check if a skill is a university/college name."""
    skill_lower = skill.lower().strip()
    return any(keyword in skill_lower for keyword in UNIVERSITY_KEYWORDS)


def is_location(skill: str) -> bool:
    """Check if a skill is a location."""
    skill_lower = skill.lower().strip()
    return skill_lower in LOCATION_KEYWORDS


def is_generic_word(skill: str) -> bool:
    """Check if a skill is a generic word."""
    skill_lower = skill.lower().strip()
    return skill_lower in GENERIC_WORDS


def is_technical_skill(skill: str) -> bool:
    """Check if a skill is a technical skill based on whitelist."""
    skill_lower = skill.lower().strip()
    
    # Direct match
    if skill_lower in TECHNICAL_CATEGORIES:
        return True
    
    # Partial match for compound skills
    for tech in TECHNICAL_CATEGORIES:
        if tech in skill_lower or skill_lower in tech:
            return True
    
    # Check for common technical patterns
    tech_patterns = [
        r'.*\.js$',  # Ends with .js
        r'.*\.net$',  # Ends with .net
        r'.*sql$',  # Ends with sql
        r'.*api$',  # Ends with api
        r'.*framework$',  # Ends with framework
        r'.*db$',  # Ends with db
        r'.*cloud$',  # Ends with cloud
        r'.*ml$',  # Ends with ml (machine learning)
        r'.*ai$',  # Ends with ai
        r'.*ops$',  # Ends with ops (DevOps)
    ]
    
    for pattern in tech_patterns:
        if re.match(pattern, skill_lower, re.IGNORECASE):
            return True
    
    return False


def filter_non_technical_skills(skills: List[str]) -> List[str]:
    """
    Filter out non-technical skills from extracted list.
    Keeps only programming languages, frameworks, databases, cloud platforms, etc.
    """
    filtered = []
    
    for skill in skills:
        if not skill or not isinstance(skill, str):
            continue
        
        skill_stripped = skill.strip()
        if not skill_stripped:
            continue
        
        # Apply exclusion rules
        if is_human_language(skill_stripped):
            logger.debug(f"Filtered out human language: {skill_stripped}")
            continue
        
        if is_hobby(skill_stripped):
            logger.debug(f"Filtered out hobby: {skill_stripped}")
            continue
        
        if is_university_or_college(skill_stripped):
            logger.debug(f"Filtered out university/college: {skill_stripped}")
            continue
        
        if is_location(skill_stripped):
            logger.debug(f"Filtered out location: {skill_stripped}")
            continue
        
        if is_generic_word(skill_stripped):
            logger.debug(f"Filtered out generic word: {skill_stripped}")
            continue
        
        # Keep if it's a technical skill (whitelist)
        if is_technical_skill(skill_stripped):
            filtered.append(skill_stripped)
        else:
            # For unknown skills, apply additional heuristics.
            # Keep if it's 2-30 chars, contains letters, and satisfies at least one of:
            #   - not capitalized (likely not a proper noun/name)
            #   - is a single word (no spaces)
            #   - contains a tech-like special character
            # NOTE: parentheses are critical here — without them Python's operator
            # precedence evaluates the bare 'or' clauses independently of the 'and',
            # which lets any single-word or special-char token bypass every check.
            if (2 <= len(skill_stripped) <= 30 and
                    re.search(r'[a-zA-Z]{2,}', skill_stripped) and
                    (not skill_stripped[0].isupper() or       # not capitalized
                     ' ' not in skill_stripped or             # single word
                     any(c in skill_stripped for c in ['.', '/', '-', '_', '+', '#']))):  # tech char
                filtered.append(skill_stripped)
            else:
                logger.debug(f"Filtered out unknown non-technical: {skill_stripped}")
    
    return filtered


def verify_skills_with_llm(skills: List[str], resume_text: str) -> List[str]:
    """
    Use LLM to verify and filter extracted skills.
    Removes company names, locations, person names, and non-skills.
    Keeps only technical skills, tools, frameworks, etc.
    """
    if not skills:
        return []
    
    # If we have too many skills, limit to top 50 for LLM processing
    skills_to_verify = skills[:50] if len(skills) > 50 else skills
    
    skills_list = ", ".join(f'"{s}"' for s in skills_to_verify)
    
    prompt = f"""Review the following extracted skills from a resume.

Resume text (first 2000 chars):
{resume_text[:2000]}

Extracted skills:
{skills_list}

Remove:
- Company names
- Person names
- Universities
- Locations
- Generic words
- Department names
- Job titles
- Project names

Keep only:
- Programming Languages
- Frameworks
- Libraries
- Databases
- Cloud Platforms
- AI/ML Technologies
- DevOps Tools
- Data Science Tools
- Analytics Tools
- Business Skills
- Software Applications
- Technical Methodologies

Return JSON only:
{{"skills": ["skill1", "skill2", ...]}}"""
    
    try:
        # Try to use the existing NuExtract model from main.py
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
        from backend.main import _get_extract_model, _get_extract_tokenizer, EXTRACT_MODEL
        
        model = _get_extract_model()
        tokenizer = _get_extract_tokenizer()
        
        # Format prompt for NuExtract
        formatted_prompt = f"""<|input|>\n{prompt}\n<|output|>"""
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from result
        import json
        # Find JSON in the output
        json_match = re.search(r'\{[^}]*"skills"[^}]*\}', result, re.DOTALL)
        if json_match:
            verified_data = json.loads(json_match.group(0))
            verified_skills = verified_data.get("skills", [])
            logger.info(f"LLM verification: {len(skills)} -> {len(verified_skills)} skills")
            return verified_skills
        else:
            logger.warning("LLM verification: Could not parse JSON response, returning original skills")
            return skills
            
    except Exception as e:
        logger.warning(f"LLM verification failed: {e}, returning original skills")
        return skills


# ---------------------------------------------------------------------------
# NEW: Confidence-Based Skill Extraction System
# ---------------------------------------------------------------------------

def extract_skills_with_confidence(text: str, tech_vocab: Set[str] = None) -> List[Dict[str, any]]:
    """
    Extract skills from entire resume using confidence-based scoring.
    Prioritizes recall over precision for NuExtract-2.0-4B optimization.
    
    Returns list of dicts: {"skill": str, "confidence": float, "sources": List[str]}
    """
    if not text:
        return []
    
    # Normalize text
    tech_vocab = tech_vocab or set()
    tech_vocab_lower = {v.lower() for v in tech_vocab}
    
    # Split resume into sections
    sections = split_resume_sections(text)
    
    # Multi-word AI/ML skills to preserve
    multi_word_ai_skills = {
        "machine learning", "deep learning", "computer vision", "natural language processing",
        "prompt engineering", "large language models", "retrieval augmented generation",
        "object detection", "face recognition", "data analysis", "business intelligence",
        "artificial intelligence", "neural networks", "generative ai", "agentic ai"
    }
    
    # AI/ML technology patterns
    ai_ml_patterns = [
        r"\b(?:LLM|GPT|OpenAI|Claude|Gemini|LangChain|LangGraph|CrewAI|AutoGen|LlamaIndex)\b",
        r"\b(?:RAG|Vector Database|ChromaDB|Qdrant|Pinecone|FAISS)\b",
        r"\b(?:Prompt Engineering|MCP|Ollama|vLLM|OpenWebUI|n8n|Flowise)\b",
        r"\b(?:PydanticAI|Haystack|Hugging Face|Transformers)\b",
        r"\b(?:TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy)\b",
        r"\b(?:Matplotlib|Seaborn|Plotly|Streamlit|Gradio)\b",
        # Cloud platforms
        r"\b(?:AWS|Amazon Web Services|Azure|GCP|Google Cloud|EC2|S3|Lambda|Docker|Kubernetes)\b",
        r"\b(?:ECS|EKS|AKS|GKE|Terraform|Ansible|Chef|Puppet)\b",
        # DevOps tools
        r"\b(?:Jenkins|GitLab CI|GitHub Actions|ArgoCD|CircleCI|Travis CI)\b",
        r"\b(?:Nginx|Apache|Nginx|HAProxy|Kong)\b",
        # Databases
        r"\b(?:MongoDB|Redis|Cassandra|Elasticsearch|Solr|Neo4j|InfluxDB)\b",
        r"\b(?:Snowflake|BigQuery|Redshift|CockroachDB|Supabase)\b",
        # Frameworks
        r"\b(?:Next\.js|Nuxt\.js|Svelte|SolidJS|Qwik|Astro)\b",
        r"\b(?:NestJS|Express|FastAPI|Flask|Django|Spring Boot)\b",
        # Frontend
        r"\b(?:React|Vue|Angular|Ember|Backbone|jQuery)\b",
        r"\b(?:Tailwind|Bootstrap|Material UI|Ant Design|Chakra UI)\b",
        # Mobile
        r"\b(?:React Native|Flutter|Ionic|SwiftUI|Jetpack Compose)\b",
        r"\b(?:Kotlin|Swift|Objective-C|Dart)\b",
    ]
    
    # Context keywords that indicate a skill is being used
    context_keywords = {
        "using", "used", "with", "built with", "developed with", "implemented with",
        "created with", "designed with", "powered by", "based on", "leveraging",
        "utilizing", "utilized", "worked on", "experience in", "expert in", "skilled in"
    }
    
    # Track skill candidates with scores and sources
    skill_candidates: Dict[str, Dict[str, any]] = {}
    
    def add_skill(skill: str, section: str, score_add: int = 1):
        """Add or update a skill candidate with score and source."""
        skill_normalized = skill.strip()
        if not skill_normalized or len(skill_normalized) < 2:
            return
        
        skill_lower = skill_normalized.lower()
        
        if skill_lower not in skill_candidates:
            skill_candidates[skill_lower] = {
                "skill": skill_normalized,
                "confidence": 0,
                "sources": set(),
                "count": 0
            }
        
        skill_candidates[skill_lower]["confidence"] += score_add
        skill_candidates[skill_lower]["sources"].add(section)
        skill_candidates[skill_lower]["count"] += 1
    
    # Extract from all sections
    all_sections = {
        "skills": sections.get("skills", ""),
        "experience": sections.get("experience", ""),
        "projects": sections.get("projects", ""),
        "summary": sections.get("summary", text[:1000]),  # First 1000 chars as summary
        "education": sections.get("education", ""),
        "certifications": sections.get("certifications", "")
    }
    
    # Section weights
    section_weights = {
        "skills": 5,
        "experience": 3,
        "projects": 3,
        "summary": 2,
        "education": 1,
        "certifications": 2
    }
    
    for section_name, section_text in all_sections.items():
        if not section_text:
            continue
        
        weight = section_weights.get(section_name, 1)
        
        # 1. Extract multi-word AI/ML skills first (preserve them)
        for multi_word in multi_word_ai_skills:
            if re.search(rf"\b{re.escape(multi_word)}\b", section_text, re.IGNORECASE):
                add_skill(multi_word.title(), section_name, weight + 2)
        
        # 2. Extract using AI/ML patterns
        for pattern in ai_ml_patterns:
            for match in re.finditer(pattern, section_text, re.IGNORECASE):
                skill = match.group(0)
                add_skill(skill, section_name, weight + 1)
        
        # 3. Extract from vocabulary (as confidence booster, not hard filter)
        for vocab_skill in tech_vocab:
            pattern = rf"(?<![a-z0-9.]){re.escape(vocab_skill)}(?![a-z0-9.])"
            if re.search(pattern, section_text, re.IGNORECASE):
                add_skill(vocab_skill, section_name, weight + 1)
        
        # 4. Extract from context sentences (e.g., "using FastAPI and PostgreSQL")
        sentences = re.split(r'[.!?]+', section_text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains context keywords
            if any(keyword in sentence_lower for keyword in context_keywords):
                # Extract potential skills from this sentence
                # Look for capitalized words (likely proper nouns/tech terms)
                words = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', sentence)
                for word in words:
                    word_clean = word.strip()
                    if len(word_clean) >= 3:
                        add_skill(word_clean, section_name, weight + 1)
                # Also extract lowercase words that are in vocabulary
                for vocab_skill in tech_vocab_lower:
                    if vocab_skill in sentence_lower:
                        add_skill(vocab_skill.title(), section_name, weight + 1)
        
        # 5. Extract from common patterns (comma-separated, bullet points, etc.)
        lines = section_text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Comma-separated skills
            if "," in line:
                for part in line.split(","):
                    part = part.strip()
                    # Remove version numbers: "Python 3.9" -> "Python"
                    part = re.sub(r'\s+\d+(\.\d+)*', '', part)
                    if 2 <= len(part) <= 60 and re.search(r"[a-zA-Z]{2,}", part):
                        part = re.sub(r"[.,;:!\?\)\]\}]+$", "", part).strip()
                        if part:
                            add_skill(part, section_name, weight)
            
            # Bullet points
            elif line.startswith(("-", "•", "*", "+")):
                skill = re.sub(r"^[-•*+\s]+", "", line).strip()
                # Remove version numbers
                skill = re.sub(r'\s+\d+(\.\d+)*', '', skill)
                if 2 <= len(skill) <= 60:
                    skill = re.sub(r"[.,;:!\?\)\]\}]+$", "", skill).strip()
                    if skill:
                        add_skill(skill, section_name, weight)
            
            # Fallback: Extract any capitalized words that look like tech terms
            # This catches skills not matched by other patterns
            else:
                # Look for sequences of capitalized words (multi-word skills)
                capitalized_words = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', line)
                for cap_word in capitalized_words:
                    cap_word = cap_word.strip()
                    if 2 <= len(cap_word) <= 60:
                        # Skip if it's a common non-skill word
                        if cap_word.lower() not in {"experience", "education", "summary", "contact", "phone", "email", "address"}:
                            add_skill(cap_word, section_name, weight - 1)  # Lower weight for fallback
    
    # Add bonus for skills found multiple times
    for skill_data in skill_candidates.values():
        if skill_data["count"] > 1:
            skill_data["confidence"] += 2
    
    # Convert to list and filter by minimum confidence (>= 3)
    final_skills = []
    for skill_data in skill_candidates.values():
        if skill_data["confidence"] >= 3:
            final_skills.append({
                "skill": skill_data["skill"],
                "confidence": min(skill_data["confidence"] / 15.0, 1.0),  # Normalize to 0-1
                "sources": list(skill_data["sources"])
            })
    
    # Sort by confidence
    final_skills.sort(key=lambda x: x["confidence"], reverse=True)
    
    logger.info(f"Confidence-based extraction: Found {len(final_skills)} skills with confidence >= 3")
    return final_skills


def normalize_skill_name(skill: str) -> str:
    """
    Normalize skill names for deduplication.
    Examples: ReactJS -> React, Postgres -> PostgreSQL, JS -> JavaScript
    """
    skill_lower = skill.lower().strip()
    
    # Common normalizations
    normalizations = {
        "reactjs": "React",
        "react.js": "React",
        "nodejs": "Node.js",
        "node.js": "Node.js",
        "node": "Node.js",
        "postgres": "PostgreSQL",
        "postgresql": "PostgreSQL",
        "js": "JavaScript",
        "javascript": "JavaScript",
        "ts": "TypeScript",
        "typescript": "TypeScript",
        "py": "Python",
        "python": "Python",
        "java": "Java",
        "cpp": "C++",
        "c++": "C++",
        "c#": "C#",
        "csharp": "C#",
        "cs": "C#",
        "css": "CSS",
        "html": "HTML",
        "sql": "SQL",
        "nosql": "NoSQL",
        "mongodb": "MongoDB",
        "mysql": "MySQL",
        "redis": "Redis",
        "docker": "Docker",
        "k8s": "Kubernetes",
        "kubernetes": "Kubernetes",
        "aws": "AWS",
        "azure": "Azure",
        "gcp": "GCP",
        "git": "Git",
        "github": "GitHub",
        "gitlab": "GitLab",
        "ci/cd": "CI/CD",
        "cicd": "CI/CD",
        # Additional common duplicates
        "pandas": "Pandas",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
        "scikit-learn": "Scikit-learn",
        "sklearn": "Scikit-learn",
        "tensorflow": "TensorFlow",
        "pytorch": "PyTorch",
        "keras": "Keras",
        "opencv": "OpenCV",
        "scipy": "SciPy",
        "nltk": "NLTK",
        "spacy": "spaCy",
        "django": "Django",
        "flask": "Flask",
        "fastapi": "FastAPI",
        "spring": "Spring",
        "spring boot": "Spring Boot",
        "express": "Express",
        "express.js": "Express",
        "angular": "Angular",
        "angularjs": "Angular",
        "vue": "Vue",
        "vue.js": "Vue",
        "svelte": "Svelte",
        "next.js": "Next.js",
        "nuxt.js": "Nuxt.js",
        "bootstrap": "Bootstrap",
        "tailwind": "Tailwind",
        "tailwind css": "Tailwind CSS",
        "material ui": "Material UI",
        "ant design": "Ant Design",
        "chakra ui": "Chakra UI",
        "jquery": "jQuery",
        "asp.net": "ASP.NET",
        "laravel": "Laravel",
        "symfony": "Symfony",
        "rails": "Rails",
        "ruby on rails": "Rails",
        "php": "PHP",
        "swift": "Swift",
        "kotlin": "Kotlin",
        "dart": "Dart",
        "rust": "Rust",
        "go": "Go",
        "golang": "Go",
        "ruby": "Ruby",
        "scala": "Scala",
        "perl": "Perl",
        "r": "R",
        "matlab": "MATLAB",
        "lua": "Lua",
        "haskell": "Haskell",
        "elixir": "Elixir",
        "erlang": "Erlang",
        "clojure": "Clojure",
        "f#": "F#",
        "julia": "Julia",
        "sass": "Sass",
        "scss": "SCSS",
        "less": "Less",
        "elasticsearch": "Elasticsearch",
        "cassandra": "Cassandra",
        "dynamodb": "DynamoDB",
        "neo4j": "Neo4j",
        "influxdb": "InfluxDB",
        "timescaledb": "TimescaleDB",
        "cockroachdb": "CockroachDB",
        "mariadb": "MariaDB",
        "oracle": "Oracle",
        "sqlite": "SQLite",
        "firestore": "Firestore",
        "supabase": "Supabase",
        "jenkins": "Jenkins",
        "circleci": "CircleCI",
        "travisci": "Travis CI",
        "travis ci": "Travis CI",
        "terraform": "Terraform",
        "ansible": "Ansible",
        "chef": "Chef",
        "puppet": "Puppet",
        "nagios": "Nagios",
        "prometheus": "Prometheus",
        "grafana": "Grafana",
        "elk": "ELK",
        "elk stack": "ELK Stack",
        "nginx": "Nginx",
        "apache": "Apache",
        "datadog": "Datadog",
        "new relic": "New Relic",
        "splunk": "Splunk",
        "sentry": "Sentry",
        "rollbar": "Rollbar",
        "pagerduty": "PagerDuty",
        "npm": "npm",
        "yarn": "Yarn",
        "pnpm": "pnpm",
        "pip": "pip",
        "conda": "Conda",
        "poetry": "Poetry",
        "composer": "Composer",
        "maven": "Maven",
        "gradle": "Gradle",
        "vscode": "VS Code",
        "visual studio": "Visual Studio",
        "intellij": "IntelliJ",
        "pycharm": "PyCharm",
        "eclipse": "Eclipse",
        "xcode": "Xcode",
        "linux": "Linux",
        "ubuntu": "Ubuntu",
        "debian": "Debian",
        "centos": "CentOS",
        "rhel": "RHEL",
        "fedora": "Fedora",
        "windows": "Windows",
        "macos": "macOS",
        "unix": "Unix",
        "bsd": "BSD",
        "android": "Android",
        "ios": "iOS",
        "react native": "React Native",
        "flutter": "Flutter",
        "xamarin": "Xamarin",
        "ionic": "Ionic",
        "cordova": "Cordova",
        "rabbitmq": "RabbitMQ",
        "activemq": "ActiveMQ",
        "kafka": "Kafka",
        "rest": "REST",
        "restful": "REST",
        "graphql": "GraphQL",
        "grpc": "gRPC",
        "soap": "SOAP",
        "openapi": "OpenAPI",
        "swagger": "Swagger",
        "oauth": "OAuth",
        "jwt": "JWT",
        "saml": "SAML",
        "ldap": "LDAP",
        "airflow": "Airflow",
        "dbt": "dbt",
        "spark": "Apache Spark",
        "hadoop": "Hadoop",
        "hive": "Hive",
        "presto": "Presto",
        "trino": "Trino",
        "snowflake": "Snowflake",
        "bigquery": "BigQuery",
        "redshift": "Redshift",
        "databricks": "Databricks",
        "fivetran": "Fivetran",
        "airbyte": "Airbyte",
        "pytest": "Pytest",
        "jest": "Jest",
        "mocha": "Mocha",
        "jasmine": "Jasmine",
        "selenium": "Selenium",
        "cypress": "Cypress",
        "playwright": "Playwright",
        "junit": "JUnit",
        "testng": "TestNG",
        "robot framework": "Robot Framework",
        "appium": "Appium",
        "langchain": "LangChain",
        "langgraph": "LangGraph",
        "openai": "OpenAI",
        "claude": "Claude",
        "crewai": "CrewAI",
        "rag": "RAG",
        "hugging face": "Hugging Face",
        "transformers": "Transformers",
        "bert": "BERT",
        "gpt": "GPT",
        "llm": "LLM",
        "machine learning": "Machine Learning",
        "deep learning": "Deep Learning",
        "nlp": "NLP",
        "computer vision": "Computer Vision",
        "reinforcement learning": "Reinforcement Learning",
        "mlops": "MLOps",
        "mlflow": "MLflow",
    }
    
    return normalizations.get(skill_lower, skill.title() if skill_lower.islower() else skill)


def deduplicate_skills(skills: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Deduplicate skills using normalization.
    Keep the highest confidence version.
    """
    normalized_map = {}
    
    for skill_data in skills:
        skill = skill_data["skill"]
        # Normalize for deduplication (case-insensitive)
        normalized = normalize_skill_name(skill).lower()
        
        if normalized not in normalized_map:
            normalized_map[normalized] = skill_data
        else:
            # Keep the version with higher confidence
            if skill_data["confidence"] > normalized_map[normalized]["confidence"]:
                normalized_map[normalized] = skill_data
    
    # Return with normalized skill names (consistent casing)
    result = []
    for normalized_key, skill_data in normalized_map.items():
        # Use the normalized name for consistent casing
        normalized_name = normalize_skill_name(skill_data["skill"])
        skill_data["skill"] = normalized_name
        result.append(skill_data)
    
    return result


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
            # Combined headers (e.g., "SKILLS EDUCATION")
            "skills education", "skills & education", "technical skills education",
            "key skills education",
        }
        if norm in known:
            return True
        # Handle combined headers like "SKILLS EDUCATION" - allow if "skills" is first word
        if norm.startswith("skills ") or norm.startswith("technical skills "):
            return True
        if "skills" in norm or "competencies" in norm or "proficiencies" in norm or "technologies" in norm:
            if not any(bad in norm for bad in ("experience", "history", "objective", "summary", "profile", "projects", "work", "employment")):
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
            # If we're currently in skills section, keep skills-related sub-headers as part of skills
            # This handles cases where skills section has subsections like "LANGUAGES", "FRAMEWORKS", "TOOLS"
            if current_section == "skills" and is_skills_header(clean_line):
                # Keep this line as part of skills section (it's a sub-header)
                section_content[current_section].append(line)
            else:
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
    Excludes education-related date ranges to avoid inflating experience.
    """
    if not experience_text:
        return 0.0

    # Filter out education-related content to avoid including education dates
    education_keywords = ["education", "academic", "university", "college", "school", "bachelor", "master", "phd", "mca", "bca", "b.tech", "m.tech", "degree", "graduation"]
    lines = experience_text.split('\n')
    filtered_lines = []
    skip_section = False
    
    for line in lines:
        line_lower = line.lower()
        # Skip lines that contain education keywords
        if any(kw in line_lower for kw in education_keywords):
            skip_section = True
            continue
        # If we hit a new section header that's not education, stop skipping
        if skip_section and any(kw in line_lower for kw in ["experience", "work", "project", "employment"]):
            skip_section = False
        if not skip_section:
            filtered_lines.append(line)
    
    experience_text = '\n'.join(filtered_lines)

    total_months = 0
    # Expanded date formats to catch: "Jan 2020 - Mar 2023", "01/2020 - 05/2023", "2020 - Present", 
    # "Aug 2018 to Oct 2020", "Jan 2020 – Mar 2023", "01/01/2020 - 05/01/2023"
    date_regex = r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s/]*\d{2,4}|\b\d{1,2}[\s/]\d{2,4}|\b\d{4}\b"
    range_regex = rf"({date_regex})\s*(?:-|to|–|—)\s*({date_regex}|present|current|now|till date|ongoing)"
    
    matches = re.findall(range_regex, experience_text)
    current_date = datetime.now()
    
    def parse_date(date_str):
        date_str = date_str.lower().strip()
        if date_str in ["present", "current", "now", "till date", "ongoing"]:
            return current_date
            
        formats = ["%b %Y", "%B %Y", "%m/%Y", "%m/%y", "%Y", "%m/%d/%Y", "%d/%m/%Y"]
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
        r"(?i)\b(testing|test|cross-browser|regression|smoke|sanity|functional|manual|automation|qa|quality|assurance)\b", # Testing terms
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
    
    # Additional rejection patterns for non-skills
    company_suffixes = {"pvt", "ltd", "limited", "inc", "llc", "corporation", "technologies", "solutions", 
                      "technology", "informatics", "systems", "services", "group", "consulting", "labs",
                      "global", "international", "india", "pvt. ltd", "pvt ltd"}
    location_keywords = {"punjab", "mohali", "chandigarh", "delhi", "mumbai", "bengaluru", "bangalore", 
                       "hyderabad", "pune", "chennai", "kolkata", "india", "usa", "uk", "uae", "canada",
                       "australia", "germany", "state", "city", "district", "county"}
    generic_prefixes = {"and", "or", "with", "using", "for", "from", "at", "in", "on", "by", "the", "a", "an"}
    generic_phrases = {"other ai tools", "other tools", "tools", "software", "applications", "programs", "etc", "etc.", "and more", "etcetera"}
    page_patterns = {r"\d+\s*/\s*\d+", r"page\s*\d+", r"pg\s*\d+"}
    
    def is_valid_skill(skill: str) -> bool:
        """Check if a candidate is actually a skill, not a company/location/generic phrase."""
        skill_lower = skill.lower().strip()
        
        # Reject if starts with generic prefix
        for prefix in generic_prefixes:
            if skill_lower.startswith(prefix + " "):
                return False
        
        # Reject if contains company suffixes
        for suffix in company_suffixes:
            if suffix in skill_lower:
                return False
        
        # Reject if it's a location
        for loc in location_keywords:
            if skill_lower == loc or loc in skill_lower:
                return False
        
        # Reject if it's just "And" followed by something
        if skill_lower.startswith("and ") or skill_lower.endswith(" and"):
            return False
        
        # Reject generic phrases
        for phrase in generic_phrases:
            if phrase in skill_lower:
                return False
        
        # Reject page numbers
        for pattern in page_patterns:
            if re.search(pattern, skill_lower):
                return False
        
        # Reject if it's a very short word that's not a known tech term
        if len(skill) <= 2 and skill_lower not in {"ai", "ar", "vr", "ml", "db", "os", "ui", "ux", "qa", "hr", "it"}:
            return False
        
        return True
    
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
            
            # Pattern 1: Comma-separated list (lenient but must be in actual resume text)
            if "," in line:
                for part in line.split(","):
                    part = part.strip()
                    # Must be 2-60 chars, check for letters
                    if 2 <= len(part) <= 60 and re.search(r"[a-zA-Z]{2,}", part):
                        # Remove common weak phrases
                        if not any(phrase in part.lower() for phrase in weak_phrases):
                            # Remove trailing punctuation
                            part = re.sub(r"[.,;:!\?\)\]\}]+$", "", part).strip()
                            if part and is_valid_skill(part):
                                skills.append(part)
            
            # Pattern 2: Bullet points (lenient but must be in actual resume text)
            elif line.startswith(("-", "•", "*", "+", ">", "o")) or re.match(r"^\d+[\).\]]", line):
                skill = re.sub(r"^[-•*+\d.\)\]>o\s]+", "", line).strip()
                if 2 <= len(skill) <= 60 and re.search(r"[a-zA-Z]{2,}", skill):
                    if not any(phrase in skill.lower() for phrase in weak_phrases):
                        skill = re.sub(r"[.,;:!\?\)\]\}]+$", "", skill).strip()
                        if skill and is_valid_skill(skill):
                            skills.append(skill)
            
            # Pattern 3: Colon-separated (Category: skill1, skill2)
            elif ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    skills_part = parts[1].strip()
                    for skill in re.split(r",\s*|\s+\|\s+", skills_part):
                        skill = skill.strip()
                        if 2 <= len(skill) <= 60 and re.search(r"[a-zA-Z]{2,}", skill):
                            if not any(phrase in skill.lower() for phrase in weak_phrases):
                                skill = re.sub(r"[.,;:!\?\)\]\}]+$", "", skill).strip()
                                if skill and is_valid_skill(skill):
                                    skills.append(skill)
            
            # Pattern 4: Pipe-separated (skill1 | skill2 | skill3) - only if multiple pipes
            elif "|" in line and line.count("|") >= 2:
                for skill in line.split("|"):
                    skill = skill.strip()
                    if 2 <= len(skill) <= 60 and re.search(r"[a-zA-Z]{2,}", skill):
                        if not any(phrase in skill.lower() for phrase in weak_phrases):
                            skill = re.sub(r"[.,;:!\?\)\]\}]+$", "", skill).strip()
                            if skill and is_valid_skill(skill):
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
                    # Apply validation filter
                    if is_valid_skill(skill):
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
                    # Apply validation filter
                    if is_valid_skill(s_clean):
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
                    # Apply validation filter
                    if is_valid_skill(p_clean):
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
                # Apply validation filter
                if is_valid_skill(cleaned):
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
            # Also try pattern-based extraction for skills not in vocabulary (only in skills section)
            extracted_skills.extend(extract_by_patterns(skills_section))
    else:
        # No dedicated skills section found - fallback to vocab matching in full text
        # but limit pattern-based extraction to avoid vocabulary dumping
        extracted_skills.extend(process_text_vocab(full_text))
        # Only apply pattern extraction if the text is relatively short (likely a simple resume)
        # to avoid extracting from long documents that might contain vocabulary lists
        if len(full_text) < 5000:
            extracted_skills.extend(extract_by_patterns(full_text))
        
    # Format & Deduplicate with spelling normalization
    final_skills = []
    seen = set()
    for s in extracted_skills:
        # Filter out invalid skills (companies, locations, generic phrases)
        if not is_valid_skill(s):
            continue
        
        # Split combined skills like "Adobe Photoshop Adobe Illustrator" into separate skills
        # If the skill contains multiple known tech terms, split them
        parts = re.split(r'\s+', s)
        if len(parts) > 2:
            # Try to split into individual skills by checking each part against vocab
            split_skills = []
            current_skill = []
            for part in parts:
                part_lower = part.lower()
                if part_lower in tech_vocab_lower or len(current_skill) == 0:
                    current_skill.append(part)
                else:
                    if current_skill:
                        split_skills.append(" ".join(current_skill))
                    current_skill = [part]
            if current_skill:
                split_skills.append(" ".join(current_skill))
            
            # If we successfully split into multiple skills, use them
            if len(split_skills) > 1:
                for split_skill in split_skills:
                    if is_valid_skill(split_skill):
                        # Normalize spelling
                        normalized = _normalize_skill_spelling(split_skill)
                        # If it's a short technical acronym, uppercase it; otherwise, title-case or preserve case
                        if len(normalized) <= 3:
                            norm = normalized.upper()
                        else:
                            norm = normalized.title() if normalized.islower() or normalized.isupper() else normalized
                        if norm.lower() not in seen:
                            seen.add(norm.lower())
                            final_skills.append(norm)
                continue
        
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
        "unknown", "anonymous", "n/a", "none",
        # Job title keywords
        "full", "stack", "front-end", "frontend", "backend", "back-end",
        "senior", "junior", "intern", "trainee", "assistant",
        # App names and watermarks
        "camscanner", "adobe", "acrobat", "pdf", "scanner", "scan",
        "docscan", "tiny scanner", "scanner app", "genius scan",
        "office lens", "camscanner pdf", "camscanner scanner",
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
    # Name was found outside the top 3 lines — keep confidence low so that
    # section headers, job titles, or company names that accidentally match
    # a 2-4 word pattern don't get promoted as the candidate's name.
    return 0.45


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
        # Take the largest single mention rather than summing all.
        # Summing inflates experience when the same (or similar) duration
        # appears in both the summary blurb and the role header.
        return round(max(month_values) / 12.0, 2)

    # 2. Look for year mentions: (\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b
    year_values: list[float] = []
    seen_years: set[float] = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", text_low):
        val = float(m.group(1))
        if 0 < val < 40 and val not in seen_years:
            seen_years.add(val)
            year_values.append(val)

    if year_values:
        # Take the largest single year mention for the same reason as above.
        # If the resume genuinely lists multiple distinct periods the
        # date-range parser (extract_experience_years) handles the summation
        # more accurately; this function is only a last-resort text scan.
        return round(max(year_values), 2)

    return 0.0


def deterministic_extract_pipeline(text: str, tech_vocab: set | None = None) -> dict:
    """
    Deterministic resume extraction used as a fallback/override for noisy PDFs.
    Returns a small, stable dict with the fields main.py expects.
    Now uses confidence-based skill extraction for better recall.
    """
    text = normalize_multi_column(text or "")
    sections = split_resume_sections(text)

    name = extract_deterministic_name(text)
    name_conf = _name_confidence_from_header(text, name)
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"V3 Extraction - Name: '{name}', Confidence: {name_conf}")
    logger.info(f"V3 Extraction - Skills section length: {len(sections.get('skills', ''))}")
    logger.info(f"V3 Extraction - Experience section length: {len(sections.get('experience', ''))}")

    vocab = set(tech_vocab or set())
    vocab.update(_build_creative_skill_vocab())

    # STRICT RULE: ONLY extract skills from explicit Skills section headers
    # If no Skills section exists, return empty skills list (do not guess from experience/projects)
    skills_section = sections.get("skills", "")
    if skills_section:
        skills_with_confidence = extract_skills_with_confidence(skills_section, vocab)
    else:
        # STRICT: No Skills section found - return empty list
        logger.info("V3 Extraction - No Skills section found, returning empty skills list")
        skills_with_confidence = []
    
    # Deduplicate skills
    skills_with_confidence = deduplicate_skills(skills_with_confidence)
    
    # Extract just the skill names (for backward compatibility)
    skills = [s["skill"] for s in skills_with_confidence]
    
    # Apply technical skill filtering to remove non-technical skills
    skills_before_filter = len(skills)
    skills = filter_non_technical_skills(skills)
    skills_after_filter = len(skills)
    
    if skills_before_filter != skills_after_filter:
        logger.info(f"V3 Extraction - Filtered {skills_before_filter - skills_after_filter} non-technical skills")
    
    logger.info(f"V3 Extraction - Skills extracted (confidence-based): {skills[:10]}...")
    logger.info(f"V3 Extraction - Total skills found: {len(skills)}")

    experience_text = sections.get("experience", "") or sections.get("projects", "")
    experience_years = extract_experience_years(experience_text)
    if experience_years == 0.0:
        # Prefer the dedicated experience block so a summary mention and the
        # role heading don't get counted twice. Fall back to the full document
        # only if the section itself doesn't contain an explicit duration.
        experience_years = extract_explicit_experience(experience_text)
    if experience_years == 0.0:
        experience_years = extract_explicit_experience(text)
    logger.info(f"V3 Extraction - Experience years: {experience_years}")

    return {
        "name": name,
        "name_conf": name_conf,
        "skills": skills,
        "skills_with_confidence": skills_with_confidence,  # New field with confidence data
        "experience_years": experience_years,
        "raw_skills_text": skills_section,  # Raw unprocessed Skills section text for UI display
    }
