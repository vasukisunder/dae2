"""
Data preparation script for DAE question similarity model.
Creates training pairs for fine-tuning a sentence transformer model.
"""
import json
import random
from itertools import combinations

def load_questions(filepath='data/dae_questions.json'):
    """Load DAE questions from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    return questions

def create_training_pairs(questions, num_negative_samples=3):
    """
    Create training pairs for sentence similarity.
    
    Strategy:
    - Positive pairs: Each question paired with itself (similarity = 1.0)
    - Negative pairs: Random question pairs (similarity = 0.0)
    
    For fine-tuning, we need triplets or pairs with similarity scores.
    """
    pairs = []
    
    # Create positive pairs (each question with itself and slight variations)
    for q in questions:
        pairs.append({
            'sentence1': q,
            'sentence2': q,
            'label': 1.0  # Perfect match
        })
    
    # Create negative pairs (random different questions)
    for _ in range(len(questions) * num_negative_samples):
        q1, q2 = random.sample(questions, 2)
        pairs.append({
            'sentence1': q1,
            'sentence2': q2,
            'label': 0.0  # Assumed dissimilar
        })
    
    # Shuffle the pairs
    random.shuffle(pairs)
    
    return pairs

def save_training_data(pairs, output_path='data/training_pairs.json'):
    """Save training pairs to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} training pairs to {output_path}")

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load questions
    print("Loading questions...")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")
    
    # Create training pairs
    print("Creating training pairs...")
    pairs = create_training_pairs(questions, num_negative_samples=2)
    
    # Save training data
    save_training_data(pairs)
    
    print("\nData preparation complete!")
    print(f"Total pairs: {len(pairs)}")
    print(f"Positive pairs: {sum(1 for p in pairs if p['label'] == 1.0)}")
    print(f"Negative pairs: {sum(1 for p in pairs if p['label'] == 0.0)}")

