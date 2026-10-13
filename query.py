# query.py
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# ---- CONFIGURATION ----
CHROMA_DB_PATH = "chroma_db"
RETRIEVE_K = 5   
DISPLAY_SOURCES_K = 2    
LLM_MODEL = "deepseek-r1:8b"

# ---- INITIALIZE ----
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection("documents")

print(f"Ready! ({collection.count()} chunks loaded)\n")


def format_source(metadata):
    source_file = metadata.get('source_file', 'Unknown')
    page_num = metadata.get('page_number', 0)
    slide_num = metadata.get('slide_number', '')
    
    if slide_num:
        return f"{source_file} (Slide {slide_num})"
    elif page_num and page_num > 0:
        return f"{source_file} (Page {page_num})"
    else:
        return source_file


def get_top_distinct_sources(metadatas, max_sources=DISPLAY_SOURCES_K):
    """Get top N distinct sources, preserving relevance order (ChromaDB returns results ranked by relevance)"""
    seen = set()
    distinct_sources = []
    
    for metadata in metadatas:
        source = format_source(metadata)
        if source not in seen:
            seen.add(source)
            distinct_sources.append(source)
        if len(distinct_sources) >= max_sources:
            break
    
    return distinct_sources


def query_documents(question, retrieve_k=RETRIEVE_K):
    question_embedding = embedder.encode([question]).tolist()
    
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=retrieve_k
    )
    
    retrieved_chunks = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]
    
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    prompt = f"""You are a helpful assistant answering questions based on company documents.

Instructions:
1. Prioritize information from the context below for any facts, figures, formulas, or specific terminology.
2. You may use general reasoning to help explain or connect ideas from the context.
3. If you provide any information NOT found in the context (e.g., general knowledge or inference), 
   clearly label it as: "[Not from provided documents: ...]"
4. If the context doesn't contain enough information to answer, say so clearly, 
   and only then optionally offer a general answer with the label above.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    answer = response['message']['content']
    
    # Get only top distinct sources (preserves relevance ranking order)
    sources = get_top_distinct_sources(retrieved_metadatas)
    
    return answer, sources


def main():
    print("="*60)
    print("RAG Chatbot - Ask questions about your documents")
    print("Type 'quit' or 'exit' to stop")
    print("="*60)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\nSearching documents...")
        answer, sources = query_documents(question)
        
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources:")
        for source in sources:
            print(f"  - {source}")


if __name__ == "__main__":
    main()