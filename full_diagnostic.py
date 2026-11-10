# full_diagnostic.py
from pypdf import PdfReader

reader = PdfReader("documents/BE OPC Capacity Planning Definition (VRFC).pdf")

print(f"Total pages: {len(reader.pages)}\n")

low_text_pages = []

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    char_count = len(text.strip())
    
    status = "Good" if char_count > 100 else "LOW TEXT"
    print(f"Page {i+1}: {char_count} characters - {status}")
    
    if char_count <= 100:
        low_text_pages.append(i+1)

print(f"\n{'='*50}")
print(f"Summary: {len(low_text_pages)} out of {len(reader.pages)} pages have low text")
print(f"Low-text pages: {low_text_pages}")