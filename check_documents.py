# check_documents.py
import os

docs_folder = "documents"
files = os.listdir(docs_folder)

print(f"Found {len(files)} file(s) in '{docs_folder}':")
for f in files:
    print(f"  - {f}")