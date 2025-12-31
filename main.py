from langchain_huggingface import HuggingFacePipeline,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from pydantic import BaseModel
from typing import List, Optional
import pdfplumber
import re
from sentence_transformers import SentenceTransformer

class ResumeSchema(BaseModel):
    name: str = ""
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = []
    experience_years: float = 0.0
    experience_summary: str = ""
    education: List[str] = []
    projects: List[str] = []
    resume_link: Optional[str] = None
    source_file: Optional[str] = None

Model = "numind/NuExtract-1.5"

token_ext = AutoTokenizer.from_pretrained(Model)
model_ext = AutoModelForCausalLM.from_pretrained(Model)

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_resume(resume_text: str):
    prompt_template = f"""You are an expert resume parser. Extract the following fields from the resume text:
Name, Email, Phone, Skills, Experience Summary, Experience Years, Education, Projects.

Provide the output in JSON format as follows:
{{
    "name": "",
    "email": "",
    "phone": "",
    "skills": [],
    "experience_years": 0.0,
    "experience_summary": "",
    "education": [],
    "projects": []
}}

Resume Text:
{resume_text}

JSON Output ONLY, nothing else:"""

    inputs = token_ext(prompt_template, return_tensors="pt", truncation=True)
    outputs = model_ext.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

    raw_output = token_ext.decode(outputs[0], skip_special_tokens=True)
    
    print("\n========== RAW LLM OUTPUT ==========")
    #print(raw_output)

    json_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_output, re.DOTALL))
    
    if json_matches:
        # Take the LAST complete JSON (the model's actual output)
        json_str = json_matches[-1].group(0)
    else:
        json_str = '{}'

    print(f"\n Found {len(json_matches)} JSON objects, using last one:")
    print("JSON STR:", json_str[:200] + "..." if len(json_str)>200 else json_str)

    # Parse with fallback
    try:
        extracted = json.loads(json_str)
        print("\n JSON PARSED SUCCESS!")
    except json.JSONDecodeError as e:
        print(f" JSON ERROR: {e}")
        extracted = {
            "name": "", "email": None, "phone": None, "skills": [],
            "experience_years": 0.0, "experience_summary": "",
            "education": [], "projects": []
        }
    
    print("\n========== FINAL EXTRACTED JSON ==========")
    print(json.dumps(extracted, indent=4))
    return extracted


embedding_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_size=384



chat_model = "Qwen/Qwen2-1.5B-Instruct"

pipe = pipeline(
    "text-generation",
    model=chat_model,
    trust_remote_code=True,
    device_map="auto",
    max_new_tokens=512,
    temperature=0.2,
)



def chatbot_answer(question: str, context: str) -> str:
    prompt = f"""You are a context-aware assistant.

Your ONLY task is to answer the user’s question using the information provided in the context below.
You MUST NOT use any external knowledge or assumptions.
You MUST NOT add information that is not present in the context.

If the answer cannot be found in the context, reply exactly with:
"Answer not found in the provided data."

Context:
{context}

Question:
{question}

Answer:
"""

    output = pipe(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.1
    )

    full_text = output[0]["generated_text"]
    return full_text.replace(prompt, "").strip()



# GLOBAL embedding model (loaded ONCE)
embeddingmodel = None

def get_embedding_model():
    global embeddingmodel
    try:
        if embeddingmodel is None:
            print(" Loading SentenceTransformer...")
            embeddingmodel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print(" Embedding model LOADED!")
        return embeddingmodel
    except Exception as e:
        print(f" Embedding model FAILED: {str(e)}")
        raise e





def extract_years_from_text(text: str) -> float:
    if not text:
        return 0.0

    match = re.search(r'(\d+(\.\d+)?)\s*\+?\s*years?', text.lower())
    if match:
        return float(match.group(1))

    return 0.0


from datetime import datetime

def parse_date(date_str: str):
    if not date_str:
        return None
    date_str = date_str.lower()
    if "present" in date_str or "current" in date_str:
        return datetime.today()

    for fmt in ("%b %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except:
            pass
    return None


import re
from datetime import datetime

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10,
    "nov": 11, "dec": 12
}

def calculate_experience_from_dates(text: str) -> float:
    if not text:
        return 0.0

    #  Normalize PDF text
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r'\s+', ' ', text)

    # Find patterns like "oct 2023", "may 2022"
    matches = re.findall(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*(\d{4})', text)

    print(" MONTH-YEAR MATCHES:", matches)

    if len(matches) < 2:
        return 0.0

    dates = []
    for month, year in matches:
        dates.append(datetime(int(year), MONTHS[month], 1))

    start_date = min(dates)
    end_date = datetime.now()

    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    return round(total_months / 12, 2)
