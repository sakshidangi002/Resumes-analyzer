import os

path = r'c:\sakshi folder\application\Resume analyzer\backend\api.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

target = '        "source": _safe_get(r, "source", "") or "",\n        "experience_line": _safe_get(r, "experience_line", "") or "",'
# Handle CRLF if present
if target not in content:
    target = target.replace('\n', '\r\n')

replacement = '        "source": _safe_get(r, "source", "") or "",\n        "source_file": r.source_file or "",\n        "experience_line": _safe_get(r, "experience_line", "") or "",'
if '\r\n' in content:
    replacement = replacement.replace('\n', '\r\n')

if target in content:
    content = content.replace(target, replacement)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fix applied successfully!")
else:
    print("Target content not found!")
