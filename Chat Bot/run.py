import uvicorn
from app.main import app

if __name__ == "__main__":
    print("🧠 Starting Wall-E Generative AI Chatbot...")
    print("📍 Mode: True Generative AI with NLP")
    print("📍 Training: From training_data/training_dataset.txt")
    print("📍 Database: MongoDB with semantic search")
    print("📍 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )