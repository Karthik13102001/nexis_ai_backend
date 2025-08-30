from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nexis-ai-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your TinyLlama endpoint from Colab
TINYLLAMA_ENDPOINT = "https://b249b1cac03c.ngrok-free.app"

@app.get("/")
def root():
    return {"message": "NEXIS AI Backend with TinyLlama!"}

@app.post("/chat")
async def chat_endpoint(request: dict):
    user_input = request.get("user_input", "")
    model = request.get("model", "tinyllama")
    files = request.get("files", [])
    
    if model == "tinyllama":
        try:
            # Send request to TinyLlama on Colab
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{TINYLLAMA_ENDPOINT}/generate",
                    json={"text": user_input, "files": files}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {"response": result.get("response", "No response")}
                else:
                    return {"response": "TinyLlama is not responding. Please check Colab."}
                    
        except Exception as e:
            return {"response": f"Error connecting to TinyLlama: {str(e)}"}
    
    else:
        # Placeholder for other models
        return {"response": f"[{model.upper()}]: Model not yet implemented"}
