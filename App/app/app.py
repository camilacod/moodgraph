from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from functools import lru_cache
import os
import httpx
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, pipeline
)

HF_MODEL = "daveni/twitter-xlm-roberta-emotion-es"
USE_REMOTE = bool(os.getenv("HF_API_TOKEN"))  # si existe el token → usa API

class InputText(BaseModel):
    text: str

app = FastAPI(title="Detector de emociones 🇪🇸")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache()         # inicializa una sola vez
def load_classifier():
    if USE_REMOTE:
        client = httpx.Client(
            headers={"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}
        )
        return client
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    return pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None)

clf = load_classifier()

@app.post("/predict")
def predict(payload: InputText):
    if not payload.text:
        raise HTTPException(400, "Texto vacío")

    # Debug log to capture the input
    print(f"Input text: {payload.text}")
    
    # Function to handle various output formats
    def normalize_output(raw_output):
        print(f"DEBUG - Raw model output: {raw_output} (type: {type(raw_output)})")
        
        # If it's already a list of dicts with label and score
        if (isinstance(raw_output, list) and 
            all(isinstance(item, dict) and 'label' in item and 'score' in item for item in raw_output)):
            return raw_output
            
        # If it's a list of lists
        if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
            raw_output = raw_output[0]  # Take the first list
        
        # If it's a direct transformers pipeline output
        if isinstance(raw_output, list):
            if raw_output and isinstance(raw_output[0], dict):
                if all('label' in item for item in raw_output):
                    # Get score field which might have various names
                    result = []
                    for item in raw_output:
                        if 'score' in item:
                            result.append({'label': item['label'], 'score': float(item['score'])})
                        elif any(k.startswith('score') for k in item):
                            score_key = next(k for k in item if k.startswith('score'))
                            result.append({'label': item['label'], 'score': float(item[score_key])})
                    if result:
                        return result
        
        # If it's a dict with label->score mapping
        if isinstance(raw_output, dict):
            return [{'label': label, 'score': float(score)} for label, score in raw_output.items()]
            
        # Final fallback
        print(f"WARNING: Could not normalize output format. Using fallback.")
        return [
            {"label": "joy", "score": 0.25}, 
            {"label": "sadness", "score": 0.25},
            {"label": "other", "score": 0.50}
        ]
    
    try:
        if USE_REMOTE:
            resp = clf.post(  # type: ignore[attr-defined]
                f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                json={"inputs": payload.text},
                timeout=30
            )
            scores = resp.json()
            print(f"API response: {scores}")
        else:
            # For local model
            raw_scores = clf(payload.text)  # type: ignore[call-arg]
            print(f"Local model raw output: {raw_scores}")
            scores = raw_scores
        
        # Normalize to standard format
        normalized_scores = normalize_output(scores)
        
        # Take top 3
        top3 = sorted(normalized_scores, key=lambda x: float(x["score"]), reverse=True)[:3]
        print(f"Top 3 emotions: {top3}")
        return {"top_3": top3}
        
    except Exception as e:
        import traceback
        print(f"Error processing prediction: {str(e)}")
        print(traceback.format_exc())
        # Return fallback response with error info
        return {"top_3": [
            {"label": "error", "score": 1.0},
            {"label": "unknown", "score": 0.0},
            {"label": "other", "score": 0.0}
        ], "error": str(e)}

# Mount the static files directory
try:
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
except Exception:
    print("Static directory not mounted - make sure the directory exists")
