import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from main import extract_text_from_pdf, extract_resume, _build_tech_skill_vocab, extract_skills_from_text, normalize_resume_text

def main():
    pdf_path = r"C:\sakshi folder\application\Resume analyzer\backend\uploads\8b4cfde2-11d7-4e25-9bed-1a229cb175f6_rashi_resume.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
        
    print("Extracting raw text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    print("--------------------------------------------------")
    print("Raw Text Length:", len(raw_text))
    print("-------------------- First 2000 Chars --------------------")
    print(raw_text[:2000])
    print("-------------------- Next 2000 Chars --------------------")
    print(raw_text[2000:4000])
    print("--------------------------------------------------")
    
    # Run deterministic vocabulary match or skills block search
    print("\nAttempting skill section extraction...")
    skills = extract_skills_from_text(raw_text)
    print("Skills extracted by extract_skills_from_text:", skills)
    
    # Run normalizer
    norm_text, _ = normalize_resume_text(raw_text)
    print("\nAttempting full V3 parser extraction...")
    # Get tech vocab
    try:
        from main import _CORE_TECH_PRIMARY
        vocab = _build_tech_skill_vocab(_CORE_TECH_PRIMARY)
        from extraction_v3 import deterministic_extract_pipeline
        extracted = deterministic_extract_pipeline(raw_text, vocab)
        print("V3 Extracted dict:", extracted)
    except Exception as e:
        print("V3 parsing error:", e)

if __name__ == "__main__":
    main()
