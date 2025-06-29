from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

# Model and tokenizer loading
MODEL_PATH = "models/mDeBERTa-v3-base-mnli-xnli/model.onnx"
TOKENIZER_PATH = "models/mDeBERTa-v3-base-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
session = ort.InferenceSession(MODEL_PATH)

# Hardcoded categories
CATEGORIES = ["Food & Drink", "Transport", "Gaming", "Other"]

def softmax(logits):
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum(axis=-1, keepdims=True)

def classify(text, labels):
    entail_scores = []
    for label in labels:
        hypothesis = f"This payment transaction is about {label}"
        tokens = tokenizer(text, hypothesis, return_tensors="np", padding=True, truncation=True)
        inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
        logits = session.run(None, inputs)[0]
        entailment_prob = softmax(logits)[0][2]  # Index 2 = entailment
        entail_scores.append(entailment_prob)
    return dict(zip(labels, entail_scores))

app = FastAPI(
    title="PiggyPal Transaction Classifier",
    description="A FastAPI wrapper for ONNX-based transaction categorization using mDeBERTa-v3",
    version="1.0.0"
)

class ClassificationRequest(BaseModel):
    transaction_text: str

class ClassificationResponse(BaseModel):
    transaction_text: str
    scores: Dict[str, float]
    predicted_category: str

@app.get("/")
async def root():
    return {"message": "PiggyPal Transaction Classifier API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "mDeBERTa-v3-base-mnli-xnli", "categories": CATEGORIES}

@app.post("/classify", response_model=ClassificationResponse)
async def classify_transaction(request: ClassificationRequest):
    try:
        if not request.transaction_text.strip():
            raise HTTPException(status_code=400, detail="Transaction text cannot be empty")
        scores = classify(request.transaction_text, CATEGORIES)
        predicted_category = min(scores.items(), key=lambda x: x[1])[0]
        return ClassificationResponse(
            transaction_text=request.transaction_text,
            scores=scores,
            predicted_category=predicted_category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/batch")
async def classify_batch_transactions(requests: List[ClassificationRequest]):
    try:
        results = []
        for request in requests:
            response = await classify_transaction(request)
            results.append(response)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")

if __name__ == "__main__":
    # CLI test
    print(classify("Nintendo CA1412771920", CATEGORIES))
    # To run the API: uvicorn app:app --reload 