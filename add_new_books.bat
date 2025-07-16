@echo off
echo Adding new books to SemanticSearch index...

REM Activate virtual environment
call env\Scripts\activate

REM Create directory for new books output
mkdir Preprocessing\new_books_output 2>nul

REM Generate summaries for new books
echo Generating summaries for new books...
python Preprocessing\generate_summaries.py --input Preprocessing\Books --output Preprocessing\new_books_output\new_books.json

REM Update the existing index
echo Updating the index...
python -c "
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load existing data
with open('Preprocessing/books_meta.json', 'r', encoding='utf-8') as f:
    existing_books = json.load(f)

# Load new books
with open('Preprocessing/new_books_output/new_books.json', 'r', encoding='utf-8') as f:
    new_books = json.load(f)

# Load model and existing index
model = SentenceTransformer('Preprocessing/all-MiniLM-L6-v2')
index = faiss.read_index('Preprocessing/books.index')

# Generate embeddings for new books
texts = [f\"{b['title']} {b['summary']}\" for b in new_books]
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Add new embeddings to index
index.add(embeddings)

# Combine metadata
all_books = existing_books + new_books

# Save updated index and metadata
faiss.write_index(index, 'Preprocessing/books.index')
with open('Preprocessing/books_meta.json', 'w', encoding='utf-8') as f:
    json.dump(all_books, f, ensure_ascii=False)

print(f'Added {len(new_books)} new books to index. Total books: {len(all_books)}')
"

echo Done! New books have been added to the index.
pause