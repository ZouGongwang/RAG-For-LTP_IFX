# ingest.py
import os
from pypdf import PdfReader
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import chromadb

# ---- CONFIGURATION ----
DOCUMENTS_FOLDER = "documents"
CHROMA_DB_PATH = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---- INITIALIZE ----
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Setting up ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    client.delete_collection("documents")
except:
    pass
collection = client.create_collection("documents")


def chunk_text(text, chunk_size=800, overlap=100):
    import re
    
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def process_pdf(filepath, filename):
    reader = PdfReader(filepath)
    all_chunks = []
    all_metadatas = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        
        if len(text.strip()) < 20:
            print(f"  ⚠️  Page {page_num + 1}: Low text content, skipping")
            continue
        
        chunks = chunk_text(text)
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "source_file": filename,
                "page_number": page_num + 1,
                "slide_number": "",  # Changed from None to empty string
                "file_path": filepath
            })
    
    return all_chunks, all_metadatas


def process_pptx(filepath, filename):
    prs = Presentation(filepath)
    all_chunks = []
    all_metadatas = []
    
    for slide_num, slide in enumerate(prs.slides):
        slide_text = []
        
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        if run.text.strip():
                            slide_text.append(run.text)
        
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                slide_text.append(f"[Notes: {notes}]")
        
        full_text = " ".join(slide_text)
        
        if len(full_text.strip()) < 20:
            print(f"  ⚠️  Slide {slide_num + 1}: Low text content, skipping")
            continue
        
        chunks = chunk_text(full_text)
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "source_file": filename,
                "page_number": 0,  # Changed from None to 0
                "slide_number": slide_num + 1,
                "file_path": filepath
            })
    
    return all_chunks, all_metadatas


def process_txt(filepath, filename):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = chunk_text(text)
    metadatas = [{
        "source_file": filename, 
        "page_number": 0,  # Changed from None to 0
        "slide_number": "",  # Changed from None to empty string
        "file_path": filepath
    } for _ in chunks]
    return chunks, metadatas


def main():
    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_counter = 0
    
    for filename in os.listdir(DOCUMENTS_FOLDER):
        filepath = os.path.join(DOCUMENTS_FOLDER, filename)
        print(f"\nProcessing: {filename}")
        
        if filename.lower().endswith('.pdf'):
            chunks, metadatas = process_pdf(filepath, filename)
        elif filename.lower().endswith('.pptx'):
            chunks, metadatas = process_pptx(filepath, filename)
        elif filename.lower().endswith('.txt'):
            chunks, metadatas = process_txt(filepath, filename)
        else:
            print(f"  Skipping unsupported file type: {filename}")
            continue
        
        print(f"  Extracted {len(chunks)} chunks")
        
        for chunk, metadata in zip(chunks, metadatas):
            all_chunks.append(chunk)
            all_metadatas.append(metadata)
            all_ids.append(f"chunk_{chunk_counter}")
            chunk_counter += 1
    
    if not all_chunks:
        print("\n❌ No chunks extracted!")
        return
    
    print(f"\n{'='*50}")
    print(f"Total chunks to embed: {len(all_chunks)}")
    print("Generating embeddings...")
    
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)
    
    print("Storing in ChromaDB...")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    print(f"\n✅ Done! {len(all_chunks)} chunks stored in ChromaDB")


if __name__ == "__main__":
    main()