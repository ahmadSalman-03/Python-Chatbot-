from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from .chatbot import chatbot
from .trainer import trainer
from .database import database

app = FastAPI(
    title="Wall-E Generative AI Chatbot",
    description="True generative AI chatbot for payment services",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: Optional[str] = "test_user"
    message: str

class ChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    is_domain_related: bool
    timestamp: str
    response_type: str

class TrainingResponse(BaseModel):
    status: str
    training_pairs: int
    message: str

@app.on_event("startup")
async def startup_event():
    """Run initial training on startup"""
    print("🔧 Starting up Generative AI Chatbot...")
    trainer.run_initial_training()

@app.get("/")
async def root():
    return {
        "message": "🧠 Wall-E Generative AI Chatbot",
        "status": "Initial training completed" if trainer.is_initial_training_done else "Training in progress",
        "mode": "Generative AI with NLP",
        "endpoints": ["POST /chat", "GET /train/status", "POST /train/start"]
    }

@app.get("/train/status")
async def training_status():
    status = trainer.get_status()
    return {
        "initial_training_done": status['initial_training_done'],
        "model_loaded": status['model_loaded'],
        "mode": "Generative AI"
    }

@app.post("/train/start")
async def start_training():
    """Manually start training"""
    success = trainer.run_initial_training()
    return TrainingResponse(
        status="success" if success else "failed",
        training_pairs=0,  # Would need to count from DB
        message="Training completed" if success else "Training failed"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not trainer.is_initial_training_done:
        raise HTTPException(status_code=400, detail="Chatbot not trained yet")
    
    try:
        # Generate response using generative AI
        response, is_domain_related = chatbot.generate_response(
            request.message, 
            request.user_id
        )
        
        # Store conversation
        conversation_id = database.store_conversation(
            request.user_id,
            request.message,
            response,
            is_domain_related
        )
        
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            conversation_id=conversation_id,
            is_domain_related=is_domain_related,
            timestamp=datetime.utcnow().isoformat(),
            response_type="generative"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "training_complete": trainer.is_initial_training_done,
        "model_loaded": chatbot.model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)