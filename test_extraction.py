import requests
import json
import requests
import json

# Read and upload a test resume
with open('backend/uploads/633c7e07-7ee6-4203-ba95-8a2907bfb381_TUSHAR_KUMAR.pdf', 'rb') as f:
    files = [('files', f)]
    response = requests.post('http://127.0.0.1:8000/upload', files=files)

if response.status_code == 200:
    result = response.json()
    print('\nRESPONSE JSON:')
    print(json.dumps(result, indent=2))
    if result and len(result) > 0:
        print("\n✓ EXTRACTION TEST RESULT:\n")
        print(f"Status: {result[0].get('status')}")
        if result[0].get('status') == 'error':
            print(f"Error Message: {result[0].get('message')}")
        else:
            print(f"Candidate Name: {result[0].get('candidate_name')}")
            if result[0].get('resume_link'):
                print(f"Resume Link: {result[0].get('resume_link')}")
    else:
        print("No result returned")
else:
    print(f'Error: {response.status_code}')
    print(response.text[:500])
