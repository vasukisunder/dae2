"""
Train a sentence similarity model on DAE questions.
This script fine-tunes a pre-trained sentence transformer model.
"""
import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

def load_training_pairs(filepath='data/training_pairs.json'):
    """Load training pairs from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    return pairs

def prepare_examples(pairs):
    """Convert pairs to InputExample objects for training."""
    examples = []
    for pair in pairs:
        examples.append(InputExample(
            texts=[pair['sentence1'], pair['sentence2']],
            label=float(pair['label'])
        ))
    return examples

def train_model(
    model_name='all-MiniLM-L6-v2',
    output_path='models/dae-similarity-model',
    epochs=4,
    batch_size=16
):
    """
    Fine-tune a sentence transformer model.
    
    Args:
        model_name: Base model to fine-tune
        output_path: Where to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Loading training data...")
    pairs = load_training_pairs()
    
    # Split into train and validation
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    
    # Prepare examples
    train_examples = prepare_examples(train_pairs)
    val_examples = prepare_examples(val_pairs)
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Use CosineSimilarityLoss for training
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Create evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples, name='dae-dev'
    )
    
    # Train the model
    print(f"\nTraining for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        evaluator=evaluator,
        evaluation_steps=500,
        warmup_steps=100,
        output_path=output_path,
        show_progress_bar=True
    )
    
    print(f"\nModel saved to {output_path}")
    return model

if __name__ == '__main__':
    print("=== DAE Question Similarity Model Training ===\n")
    
    # Train the model
    model = train_model(
        model_name='all-MiniLM-L6-v2',
        output_path='models/dae-similarity-model',
        epochs=4,
        batch_size=16
    )
    
    print("\nTraining complete!")
    print("\nYou can now use this model with the web application.")

