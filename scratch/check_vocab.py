import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from main import extract_text_from_pdf, extract_skills_from_text, _build_tech_skill_vocab, _CORE_TECH_PRIMARY, _skill_vocab_hit, _norm_header

def main():
    pdf_path = r"C:\sakshi folder\application\Resume analyzer\backend\uploads\bd11d332-5976-449a-911e-ed49ab272644_Kalpana_Raina_ATS_Resume-1.pdf"
    text = extract_text_from_pdf(pdf_path)
    
    vocab = _build_tech_skill_vocab(_CORE_TECH_PRIMARY)
    
    # Check individual terms
    test_terms = ["uipath", "UiPath", "opencv", "OpenCV", "mediapipe", "MediaPipe", 
                  "uipath orchestrator", "python", "mysql", "process automation"]
    
    print("Vocab hit check:")
    for term in test_terms:
        hit = _skill_vocab_hit(_norm_header(term), vocab)
        print(f"  '{term}' -> {'HIT' if hit else 'MISS'}")
    
    print("\nFull extracted skills:")
    skills = extract_skills_from_text(text)
    print(skills)

if __name__ == "__main__":
    main()
