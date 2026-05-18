import requests

url = "http://localhost:8000/api/auth/login"
data = {
    "username": "admin",
    "password": "password123" # try some dummy password
}
try:
    print("Sending request...")
    res = requests.post(url, json=data, timeout=5)
    print("Status:", res.status_code)
    print("Response:", res.text)
except Exception as e:
    print("Error:", e)
