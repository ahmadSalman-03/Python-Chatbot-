import requests
import time

def test_generative_chatbot():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Generative AI Chatbot...")
    print("=" * 60)
    
    # Test training status
    response = requests.get(f"{base_url}/train/status")
    print(f"📊 Training Status: {response.json()}")
    
    # Test domain-specific questions
    test_questions = [
        "How do I send money to my friend?",
        "What is offline transaction and how does it work?",
        "Is my money safe in Wall-E?",
        "How can I earn rewards?",
        "What is the temperature today?",  # Out of domain
        "Tell me about cricket matches",  # Out of domain
        "How to check my wallet balance?",
        "What are the transaction limits?"
    ]
    
    for question in test_questions:
        print(f"\n💬 User: {question}")
        
        response = requests.post(f"{base_url}/chat", json={
            "user_id": "test_user_001",
            "message": question
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 Bot: {data['response']}")
            print(f"   Domain Related: {data['is_domain_related']}")
            print(f"   Response Type: {data['response_type']}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
        
        time.sleep(1)

if __name__ == "__main__":
    test_generative_chatbot()