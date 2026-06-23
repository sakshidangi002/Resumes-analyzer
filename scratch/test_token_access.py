import requests
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "abc2025"
ALGORITHM = "HS256"

# Create token for user ID 3 (saloni), who has roles ['HR', 'Employee']
expire = datetime.utcnow() + timedelta(minutes=60)
to_encode = {
    "exp": expire,
    "sub": "3",
    "roles": ["HR", "Employee"],
    "employee_id": 6
}
token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
print("Generated token:", token)

# Test with Authorization Header
headers = {
    "Authorization": f"Bearer {token}"
}
url_api = "http://127.0.0.1:5001/resume-api/resumes"
try:
    print(f"Sending GET to {url_api} with Authorization header...")
    r = requests.get(url_api, headers=headers, timeout=5)
    print("Status code:", r.status_code)
    print("Response JSON:", r.text[:300])
except Exception as e:
    print("Error:", e)

# Test with query parameter token
url_query = f"http://127.0.0.1:5001/resume-api/resumes?token={token}"
try:
    print(f"Sending GET to {url_query} without Authorization header...")
    r = requests.get(url_query, timeout=5)
    print("Status code:", r.status_code)
    print("Response JSON:", r.text[:300])
except Exception as e:
    print("Error:", e)
