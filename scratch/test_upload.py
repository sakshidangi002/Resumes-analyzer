import os
import requests
from jose import jwt
from datetime import datetime, timedelta

# Create valid JWT token
SECRET_KEY = "abc2025"
ALGORITHM = "HS256"
to_encode = {
    "exp": datetime.utcnow() + timedelta(hours=1),
    "sub": "6", # user ID of admin in the database
    "roles": ["Admin"]
}
token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

headers = {
    "Authorization": f"Bearer {token}"
}

pdf_path = r"c:\sakshi folder\application\Resume analyzer\backend\uploads\1ab2ef86-c846-419a-83b8-fda5a8e7fe95_Imran_FSD_CV.pdf"

with open(pdf_path, 'rb') as f:
    files = [('files', ('test_resume.pdf', f, 'application/pdf'))]
    response = requests.post('http://127.0.0.1:8001/upload', files=files, headers=headers)

print(f"Status Code: {response.status_code}")
try:
    print(response.json())
except Exception:
    print(response.text)
