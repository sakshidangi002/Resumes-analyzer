import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from main import extract_text_from_pdf, extract_skills_from_text

def main():
    # Find the most recent PDF with "Progra" or "Mming" splitting issue - check uploads folder
    upload_dir = r"C:\sakshi folder\application\Resume analyzer\backend\uploads"
    pdfs = sorted([f for f in os.listdir(upload_dir) if f.endswith(".pdf")], key=lambda f: os.path.getmtime(os.path.join(upload_dir, f)), reverse=True)
    
    print("Most recent PDFs:")
    for f in pdfs[:6]:
        print(" -", f)
    
    # Try the most recent pdf files and see which one has the scrambled skills issue
    for filename in pdfs[:4]:
        pdf_path = os.path.join(upload_dir, filename)
        text = extract_text_from_pdf(pdf_path)
        
        # Look for the skills section
        lines = text.splitlines()
        in_skills = False
        skill_lines = []
        for ln in lines:
            ln_low = ln.strip().lower()
            if "skills" in ln_low or "expertise" in ln_low or "technologies" in ln_low:
                in_skills = True
                skill_lines.append(f">>> HEADER: {ln.strip()}")
                continue
            if in_skills:
                if ln.strip() and any(h in ln_low for h in ["experience", "education", "project", "summary", "certification"]):
                    break
                skill_lines.append(ln)
                if len(skill_lines) > 20:
                    break
        
        if any("progra" in sl.lower() or "mming" in sl.lower() for sl in skill_lines):
            print(f"\nFOUND PROBLEMATIC FILE: {filename}")
            print("\nSkills Section Raw Lines:")
            for sl in skill_lines:
                print(repr(sl))
            skills = extract_skills_from_text(text)
            print("\nExtracted Skills:", skills)
            break
    else:
        print("\nNot found - printing first PDF skill sections:")
        for filename in pdfs[:3]:
            pdf_path = os.path.join(upload_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            print(f"\nFile: {filename}")
            print("--- First 1500 chars ---")
            print(text[:1500])

if __name__ == "__main__":
    main()
