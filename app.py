from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import httpx
import os

# Google Gemini
import google.generativeai as genai

app = FastAPI(title="NEXIS AI Backend with Multi-Model Support", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nexis-ai-frontend.onrender.com"],  # update if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN environment variable not set.")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

GITHUB_MODELS_API = "https://models.inference.ai.azure.com"

MODEL_MAPPING = {
    "tinyllama":       "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "phi3":            "microsoft/Phi-3.5-mini-instruct",
    "smollm2":         "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen":            "gpt-4o",
    "gemini-pro":      "gemini-pro",
    "gemini-1.5-flash":"gemini-1.5-flash",
    "gemini-1.5-pro":  "gemini-1.5-pro",
    "gemini-2.5-flash":"gemini-2.5-flash"
}

class ChatRequest(BaseModel):
    user_input: str
    model: Optional[str] = "qwen"
    files: Optional[List[dict]] = []

@app.get("/")
def root():
    return {"message": "NEXIS AI Backend connected to GitHub & Gemini Models API"}

@app.post("/chat")
async def chat(request: ChatRequest):
    model_key = request.model or "qwen"
    model_name = MODEL_MAPPING.get(model_key)
    if not model_name:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    
    context = ""
    if request.files:
        context = f"\nUploaded {len(request.files)} files:\n"
        for file in request.files:
            context += f"- {file.get('name','unknown')}\n"

    # GitHub path
    if model_key in ["tinyllama", "phi3", "smollm2", "qwen"]:
        messages = [
            {"role": "system", "content": "You are NEXIS AI, a helpful and knowledgeable assistant."},
            {"role": "user", "content": request.user_input + context}
        ]
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{GITHUB_MODELS_API}/chat/completions", headers=headers, json=body
            )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"GitHub Models API error: {response.text}")
        response_json = response.json()
        ai_text = response_json["choices"][0]["message"]["content"]
        return {"response": ai_text}
    
    # Gemini path
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(model_name)
            prompt = request.user_input + context
            gemini_response = model.generate_content(prompt)
            if hasattr(gemini_response, "text"):
                return {"response": gemini_response.text}
            return {"response": "Gemini did not return text."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
