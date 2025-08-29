from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="NEXIS AI Backend", version="1.0.0")

# CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nexis-ai-frontend.onrender.com",  # Your frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    model: str = "tinyllama"
    files: Optional[List[dict]] = []

@app.get("/")
async def root():
    return {"message": "NEXIS AI Backend is running!", "status": "healthy"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests - placeholder for now"""
    user_text = request.user_input
    selected_model = request.model
    
    # Simple placeholder response (we'll add AI models in Step 3)
    response = f"[{selected_model.upper()}]: Echo - {user_text}"
    
    if request.files:
        response += f"\nFiles received: {len(request.files)} files"
    
    return {"response": response}

# For Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
