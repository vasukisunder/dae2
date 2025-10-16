"""
Flask web application for DAE Question Similarity.
Users can input their own DAE question and find similar questions from the dataset.
"""
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json
import os
import numpy as np

app = Flask(__name__)

# Global variables to store model and data
model = None
questions = []
question_embeddings = None

def load_model_and_data():
    """Load the trained model and question data."""
    global model, questions, question_embeddings
    
    print("Loading model and data...")
    
    # Check if fine-tuned model exists, otherwise use base model
    model_path = 'models/dae-similarity-model'
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        model = SentenceTransformer(model_path)
    else:
        print("Fine-tuned model not found. Using base model: all-MiniLM-L6-v2")
        print("Run 'python prepare_data.py' and 'python train_model.py' to create a fine-tuned model.")
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load questions
    with open('data/dae_questions.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Pre-compute embeddings for all questions
    print("Computing embeddings for all questions...")
    question_embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)
    
    print("Ready!")

def find_similar_questions(query, top_k=10):
    """
    Find the most similar questions to the query.
    
    Args:
        query: User's DAE question
        top_k: Number of similar questions to return
    
    Returns:
        List of (question, similarity_score) tuples
    """
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarities
    cos_scores = util.cos_sim(query_embedding, question_embeddings)[0]
    
    # Get top-k results
    top_results = np.argsort(-cos_scores.cpu().numpy())[:top_k]
    
    results = []
    for idx in top_results:
        results.append({
            'question': questions[idx],
            'similarity': float(cos_scores[idx]),
            'percentage': round(float(cos_scores[idx]) * 100, 1)
        })
    
    return results

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/similar', methods=['POST'])
def get_similar():
    """API endpoint to find similar questions."""
    data = request.json
    query = data.get('query', '').strip()
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        # Find similar questions
        results = find_similar_questions(query, top_k)
        
        return jsonify({
            'query': query,
            'results': results,
            'total_questions': len(questions)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get statistics about the dataset."""
    return jsonify({
        'total_questions': len(questions),
        'model_type': 'fine-tuned' if os.path.exists('models/dae-similarity-model') else 'base'
    })

if __name__ == '__main__':
    # Load model and data on startup
    load_model_and_data()
    
    # Run the app
    print("\n=== DAE Question Similarity Web App ===")
    print("Open your browser and go to: http://localhost:8080")
    print("Press CTRL+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=8080)

