# PiggyPal Transaction Classifier API

A FastAPI wrapper for the ONNX-based transaction categorization system using mDeBERTa-v3 model.

## Features

- **Single Transaction Classification**: Classify individual transactions into predefined categories
- **Batch Classification**: Process multiple transactions at once
- **Health Check**: Monitor API status and model availability
- **Automatic Documentation**: Interactive API docs available at `/docs`

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model Files**: Make sure the ONNX model files are present in the `models/` directory:
   - `models/mDeBERTa-v3-base-mnli-xnli/model.onnx`
   - `models/mDeBERTa-v3-base-mnli-xnli/config.json`
   - `models/mDeBERTa-v3-base-mnli-xnli/tokenizer.json`
   - And other required model files

## Running the API

### Development Server
```bash
python app.py
```

### Production Server (with uvicorn)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
- **GET** `/health`
- Returns API status and model information

### 2. Single Classification
- **POST** `/classify`
- **Request Body**:
  ```json
  {
    "transaction_text": "Nintendo CA1412771920",
    "categories": ["Food & Drink", "Transport", "Groceries", "Other"]
  }
  ```
- **Response**:
  ```json
  {
    "transaction_text": "Nintendo CA1412771920",
    "categories": ["Food & Drink", "Transport", "Groceries", "Other"],
    "scores": {
      "Food & Drink": 0.1,
      "Transport": 0.05,
      "Groceries": 0.02,
      "Other": 0.83
    },
    "predicted_category": "Other"
  }
  ```

### 3. Batch Classification
- **POST** `/classify/batch`
- **Request Body**: Array of classification requests
- **Response**: Array of classification responses

## API Documentation

- **Interactive API Docs**: `http://localhost:8000/docs` (Swagger UI)

## Example Usage

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Single classification
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_text": "Uber *TRIP",
       "categories": ["Food & Drink", "Transport", "Groceries", "Other"]
     }'
```


## Performance

- The ONNX model provides fast inference times (~60ms for batch categorization)
- Model is loaded once at startup for optimal performance 