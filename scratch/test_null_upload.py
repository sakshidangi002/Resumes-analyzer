import os
import requests
from jose import jwt

def main():
    SECRET_KEY = "abc2025"
    ALGORITHM = "HS256"
    
    # Generate JWT token for admin (user_id 6)
    token = jwt.encode({"sub": "6"}, SECRET_KEY, algorithm=ALGORITHM)
    
    url = "http://127.0.0.1:8001/upload"
    pdf_path = r"C:\sakshi folder\application\Resume analyzer\backend\uploads\ee5b4ff4-d174-4a81-918d-f0c7e452ca95_Navjotkaur_resume__1_.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
        
    print(f"Uploading {pdf_path}...")
    headers = {"Authorization": f"Bearer {token}"}
    with open(pdf_path, "rb") as f:
        files = {"files": (os.path.basename(pdf_path), f, "application/pdf")}
        response = requests.post(url, files=files, headers=headers)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    main()
