


## How It Works

### 1. Sentence Embeddings

The application uses sentence-transformers to convert questions into 384-dimensional vectors (embeddings) that capture semantic meaning.

### 2. Cosine Similarity

When you enter a question, it:
1. Converts your question to an embedding
2. Compares it to all pre-computed question embeddings using cosine similarity
3. Returns the most similar questions ranked by score

### 3. Model Architecture

- **Base**: `all-MiniLM-L6-v2` (22M parameters)
- **Output**: 384-dimensional dense vectors
- **Training**: Contrastive learning with cosine similarity loss
- **Inference**: Fast semantic search using vector similarity

## API Endpoints

### POST `/api/similar`

Find similar questions to a query.

**Request**:
```json
{
  "query": "feel anxious in social situations?",
  "top_k": 10
}
```

**Response**:
```json
{
  "query": "feel anxious in social situations?",
  "total_questions": 639,
  "results": [
    {
      "question": "get nervous being out in public anymore?",
      "similarity": 0.8543,
      "percentage": 85.4
    },
    ...
  ]
}
```

### GET `/api/stats`

Get dataset statistics.

**Response**:
```json
{
  "total_questions": 639,
  "model_type": "fine-tuned"
}
```

## Customization

### Adjust Number of Training Pairs

Edit `prepare_data.py`:
```python
pairs = create_training_pairs(questions, num_negative_samples=2)  # Increase for more data
```

### Change Training Parameters

Edit `train_model.py`:
```python
model = train_model(
    model_name='all-MiniLM-L6-v2',  # Try different models
    output_path='models/dae-similarity-model',
    epochs=4,  # More epochs for better training
    batch_size=16  # Adjust based on your GPU memory
)
```

### Use a Different Base Model

Replace `'all-MiniLM-L6-v2'` with:
- `'all-mpnet-base-v2'` - More accurate but slower (768 dimensions)
- `'paraphrase-MiniLM-L3-v2'` - Faster but less accurate (384 dimensions)
- Any model from [Sentence-Transformers](https://www.sbert.net/docs/pretrained_models.html)

## Performance

- **Embedding generation**: ~100ms for a single question
- **Similarity search**: <10ms across 639 questions
- **Total query time**: ~110ms end-to-end
- **Memory usage**: ~500MB (model + embeddings)

## Troubleshooting

### PyTorch Installation Issues

If PyTorch installation fails, install it separately:
```bash
# CPU only (faster install)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install other requirements
pip install -r requirements.txt
```

### Model Download Fails

If model download is slow or fails:
1. Check your internet connection
2. The model (~90MB) will be cached after first download
3. Manually download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

### Port Already in Use

If port 5000 is busy, change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Change to any free port
```

## Technologies Used

- **Backend**: Flask (Python web framework)
- **ML Model**: Sentence-Transformers (SBERT)
- **Deep Learning**: PyTorch
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Dataset**: 639 DAE questions from Reddit

## Future Enhancements

- [ ] Add question clustering to discover themes
- [ ] Implement user voting on similarity accuracy
- [ ] Add ability to submit new questions to the dataset
- [ ] Create API rate limiting for production use
- [ ] Add multilingual support
- [ ] Implement caching for common queries
- [ ] Add authentication for personalized experience

## License

This project is for educational and personal use.

## Contributing

Feel free to fork this project and submit pull requests for improvements!

## Contact

For questions or feedback, please open an issue on the repository.

---

**Made with ❤️ using Sentence-Transformers and Flask**

