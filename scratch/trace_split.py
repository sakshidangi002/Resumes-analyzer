import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from main import (extract_text_from_pdf, _is_skills_header_line, _norm_header,
                  _looks_like_header, _tokenize_skill_candidates, _collapse_whitespace)

def main():
    pdf_path = r"C:\sakshi folder\application\Resume analyzer\backend\uploads\bd11d332-5976-449a-911e-ed49ab272644_Kalpana_Raina_ATS_Resume-1.pdf"
    text = extract_text_from_pdf(pdf_path)
    
    # Step 1: Simulate section collection
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    section_blocks = []
    i = 0
    n = len(lines)
    stop_headers = {
        "experience","work experience","work history","professional experience",
        "projects","project experience","education","summary","profile","objective",
        "certifications","achievements","awards","languages",
    }
    while i < n:
        ln = lines[i]
        if _is_skills_header_line(ln):
            start = i + 1
            section = []
            blank_streak = 0
            for j in range(start, n):
                cur = lines[j]
                if not cur:
                    blank_streak += 1
                    if section and blank_streak >= 2:
                        break
                    continue
                blank_streak = 0
                if _norm_header(cur) in stop_headers and section:
                    break
                if _looks_like_header(cur) and section:
                    break
                section.append(cur)
                if len(section) >= 20:
                    break
            if section:
                section_blocks.append(section)
            i = j
        else:
            i += 1

    print("Section blocks found:", len(section_blocks))
    for idx, block in enumerate(section_blocks):
        print(f"  Block {idx}:", block)

    if not section_blocks:
        print("NO SECTION BLOCKS FOUND!")
        return

    # Step 2: Join and apply preprocessing
    section_joined = "\n".join(["\n".join(sec) for sec in section_blocks if sec]).strip()
    print("\nRaw section_joined:")
    print(repr(section_joined))

    # Fix hyphen-broken words
    section_joined = re.sub(r"-\n(\S)", r"\1", section_joined)

    # Fix mid-word spaces
    def _fix_broken_word(m):
        left, right = m.group(1), m.group(2)
        standalone = {"in","an","of","to","at","by","or","as","is","it","be","on","up","no","so","do","go","my","we","us","for","the","and","are","but","not","was","has","had","its","can","may","one","two","our","via","per","data","open","code","core","java","html","node","next","pipe","time","user","base","work","word"}
        if left.lower() in standalone or right.lower() in standalone:
            return m.group(0)
        if right.lower().startswith(('mm','nd','tt','ss','pp','ll','rr','cc')) or any(right.lower().startswith(s) for s in ('mming','tion','ning','ring','ling','king')):
            return left + right
        return m.group(0)
    section_joined = re.sub(r'([a-zA-Z]{2,6}) ([a-z]{2,6})\b', _fix_broken_word, section_joined)

    # Colon processing
    processed_lines = []
    for raw_ln in section_joined.splitlines():
        stripped = raw_ln.strip()
        colon_match = re.match(r'^([^:]{2,50}):\s*(.+)$', stripped)
        if colon_match:
            processed_lines.append(colon_match.group(2).strip())
            print(f"  Colon match: '{stripped}' -> '{colon_match.group(2).strip()}'")
        else:
            processed_lines.append(stripped)
    section_text = "\n".join(processed_lines)

    print("\nFinal section_text after preprocessing:")
    print(repr(section_text))

    # Step 3: Check what _tokenize_skill_candidates produces
    print("\nTokenized candidates:")
    for t in _tokenize_skill_candidates(section_text):
        print(f"  {repr(t)}")

if __name__ == "__main__":
    main()
