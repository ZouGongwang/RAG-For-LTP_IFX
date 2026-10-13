# test_ollama.py
import ollama

response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Say hello in 5 words or less'}
])

print("Ollama response:", response['message']['content'])