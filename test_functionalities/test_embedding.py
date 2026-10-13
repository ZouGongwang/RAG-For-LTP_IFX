# test_embedding.py
from sentence_transformers import SentenceTransformer

print("Downloading/loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Quick test
sentences = ["Hello world", "How are you?"]
embeddings = model.encode(sentences)

print(f"✅ Success! Embedding shape: {embeddings.shape}")
print(f"First embedding (first 5 values): {embeddings[0][:5]}")