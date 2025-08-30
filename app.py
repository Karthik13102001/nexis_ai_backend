from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nexis-ai-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "NEXIS AI Backend is running!"}

@app.post("/chat")
def chat_endpoint(request: dict):
    user_input = request.get("user_input", "")
    model = request.get("model", "tinyllama")
    files = request.get("files", [])
    
    response = f"[{model.upper()}]: Echo - {user_input}"
    
    if files:
        response += f"\nFiles received: {len(files)} files"
    
    return {"response": response}
