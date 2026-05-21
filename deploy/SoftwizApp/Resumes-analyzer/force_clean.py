"""Force delete all chromadb files and directories"""
import shutil
import os
import glob

base = "."

print("[*] Forcefully removing all chromadb artifacts...")

# Remove specific directories
for pattern in ["chromadb", "chroma_db", "*chromadb*backup*"]:
    for path in glob.glob(os.path.join(base, pattern)):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                print(f"✓ Removed directory: {path}")
            elif os.path.isfile(path):
                os.remove(path)
                print(f"✓ Removed file: {path}")
        except Exception as e:
            print(f"✗ Failed on {path}: {e}")

print("\n[✓] All chromadb files removed. Server will recreate fresh schema on startup.")
