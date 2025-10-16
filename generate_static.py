"""
Generate static embeddings for the static website.
Pre-computes embeddings so the static site doesn't need Python.
"""
import json
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading questions...")
with open('data/dae_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

print(f"Computing embeddings for {len(questions)} questions...")
embeddings = model.encode(questions, show_progress_bar=True)

# Convert to list for JSON serialization
embeddings_list = embeddings.tolist()

# Create static data file
static_data = {
    'questions': questions,
    'embeddings': embeddings_list
}

print("Saving to docs/data.json...")
with open('docs/data.json', 'w', encoding='utf-8') as f:
    json.dump(static_data, f)

print(f"✓ Generated static data file: {len(static_data['questions'])} questions")
print(f"✓ File size: ~{len(json.dumps(static_data)) / 1024 / 1024:.1f}MB")
print("\nYou can now deploy the docs/ folder to GitHub Pages!")

