from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import pipeline
import torchaudio
import torch
import numpy as np
import shutil
import os
import tempfile
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load emotion models globally
anger_classifier = None
other_classifier = None

# Gender classifier (simulated with logistic regression)
gender_classifier = None
scaler = StandardScaler()

# ------------- Emotion Model Loader -------------

@app.on_event("startup")
async def load_models():
    global anger_classifier, other_classifier, gender_classifier
    try:
        logger.info("Loading emotion models...")
        anger_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        other_classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=None
        )
        # Simulate gender classifier training (replace with real training data)
        logger.info("Initializing gender classifier...")
        # Dummy training data: 100 samples, 13 MFCC features
        X_train = np.random.rand(100, 13)  # Replace with real MFCC features
        y_train = np.random.choice([0, 1], 100)  # 0: Female, 1: Male
        gender_classifier = LogisticRegression().fit(scaler.fit_transform(X_train), y_train)
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load models")

# ------------- Utility Functions -------------

def clean_text(text: str) -> str:
    """Clean text by removing filler words and normalizing."""
    text = re.sub(r'\b(um|uh|like|you know)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def is_anger_query(text: str) -> bool:
    """Check if text contains anger-related keywords."""
    anger_keywords = ["unacceptable", "worst", "angry", "frustrated", "tired", "ridiculous", "annoying"]
    return any(word in text.lower() for word in anger_keywords)

# ------------- Emotion Prediction -------------

class TextInput(BaseModel):
    text: str

def predict_emotion(text: str):
    """Predict emotion using ensemble of two models."""
    try:
        text = clean_text(text)
        if not text:
            raise ValueError("Cleaned text is empty")

        # Check for anger keywords first
        if is_anger_query(text):
            logger.info("Anger detected via keywords")
            return "Anger", 0.9

        # Get predictions from both models
        anger_scores = anger_classifier(text)[0]
        anger_score = next((item['score'] for item in anger_scores if item['label'].lower() == 'anger'), 0.0)
        
        if anger_score > 0.6:
            logger.info(f"Anger detected with score {anger_score}")
            return "Anger", anger_score

        other_scores = other_classifier(text)[0]
        best = max(other_scores, key=lambda x: x['score'])
        emotion = best['label'].replace('emotion:', '').capitalize()
        confidence = best['score']

        # Ensemble: average confidence if emotions match
        other_emotion = max(anger_scores, key=lambda x: x['score'])['label'].capitalize()
        if other_emotion == emotion:
            confidence = (confidence + max(anger_scores, key=lambda x: x['score'])['score']) / 2

        logger.info(f"Predicted emotion: {emotion} with confidence {confidence}")
        return emotion, confidence
    except Exception as e:
        logger.error(f"Emotion prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Emotion prediction failed: {str(e)}")

@app.post("/predict_emotion")
async def predict_emotion_api(input: TextInput):
    """API endpoint for emotion prediction."""
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    emotion, confidence = predict_emotion(input.text)
    return {
        "status": "success",
        "text": input.text,
        "emotion": emotion,
        "confidence": round(confidence, 4)
    }

# ------------- Gender Prediction -------------

def extract_features(file_path: str) -> np.ndarray:
    """Extract MFCC features from audio file."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:  # Resample to 16kHz for consistency
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13)(waveform)
        return mfcc.mean(dim=2).squeeze().numpy()
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

@app.post("/predict_gender")
async def predict_gender(file: UploadFile = File(...)):
    """API endpoint for gender prediction from audio."""
    # Validate file extension
    valid_extensions = {'.wav', '.mp3'}
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in valid_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Use {valid_extensions}")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_file_path = tmp.name

    try:
        # Extract features
        features = extract_features(temp_file_path)
        # Scale features
        features_scaled = scaler.transform([features])
        # Predict gender
        gender_pred = gender_classifier.predict(features_scaled)[0]
        confidence = gender_classifier.predict_proba(features_scaled)[0][gender_pred]
        gender = "Female" if gender_pred == 0 else "Male"

        logger.info(f"Predicted gender: {gender} with confidence {confidence}")
        return JSONResponse(
            content={
                "status": "success",
                "predicted_gender": gender,
                "confidence": round(confidence, 4)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gender prediction failed: {str(e)}")
    finally:
        os.remove(temp_file_path)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Emotion and Gender Prediction API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)