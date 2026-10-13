# test_vision_extraction.py
from pdf2image import convert_from_path
import ollama
import os

POPPLER_PATH = r"C:\Users\zougo\Downloads\Poppler(Windows)\poppler-26.09.0\Library\bin"  

def test_diagram_extraction(pdf_path, page_num=0, model='llava:7b'):
    images = convert_from_path(
        pdf_path, 
        first_page=page_num+1, 
        last_page=page_num+1,
        poppler_path=POPPLER_PATH
    )
    image = images[0]
    
    temp_image_path = "temp_page.png"
    image.save(temp_image_path)
    
    print(f"Analyzing page {page_num+1} of {pdf_path} using {model}...\n")
    
    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': '''Describe this document page in detail. 
            If there are diagrams, charts, or graphs, explain what data/information 
            they show, including any numbers, labels, trends, or relationships depicted. 
            If there is text, include the key points. Be thorough and specific about 
            any data values, axis labels, categories, or connections shown.''',
            'images': [temp_image_path]
        }]
    )
    
    print("VISION MODEL OUTPUT:")
    print(response['message']['content'])
    print("\n" + "="*60 + "\n")
    
    os.remove(temp_image_path)
    return response['message']['content']

# Make sure filename EXACTLY matches your file, including .pdf extension
test_diagram_extraction("documents/BE OPC Capacity Planning Definition (VRFC).pdf", page_num=10)