path = r"c:\sakshi folder\application\Resume analyzer\backend\api.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Find the return statement specifically
search_idx = content.find("@app.get(\"/resumes/search\"")
if search_idx > -1:
    snippet = content[search_idx:search_idx+7000]
    ret_idx = snippet.find("return {")
    if ret_idx > -1:
        print("=== Return dict of search_resumes ===")
        print(snippet[ret_idx:ret_idx+600])
