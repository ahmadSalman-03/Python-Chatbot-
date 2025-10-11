# app/test_conversation.py
import requests
import time
import sys
import os

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_complete_conversation():
    base_url = "http://localhost:8000"
    user_id = "test_user_001"
    
    print("💬 Testing Complete Conversation Flow...")
    print("=" * 60)
    
    conversation_flow = [
        "Hello!",
        "How do I send money to my friend?",
        "What about offline payments?",
        "Is it secure?",
        "Thank you for your help!",
        "Goodbye!"
    ]
    
    for message in conversation_flow:
        print(f"\n👤 User: {message}")
        
        try:
            response = requests.post(f"{base_url}/chat", json={
                "user_id": user_id,
                "message": message
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"🤖 Wall-E: {data['response']}")
                
                # Show conversation type
                if 'hello' in message.lower():
                    conv_type = "Greeting"
                elif 'bye' in message.lower() or 'goodbye' in message.lower():
                    conv_type = "Farewell"
                elif 'thank' in message.lower():
                    conv_type = "Thanks"
                else:
                    conv_type = "Domain Question"
                    
                print(f"   💡 Type: {conv_type} | Domain Related: {data['is_domain_related']}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server. Make sure the chatbot is running on port 8000!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
        
        time.sleep(1)

def test_mixed_conversation():
    base_url = "http://localhost:8000"
    
    print("\n🎭 Testing Mixed Conversation (Domain + General)...")
    print("=" * 60)
    
    test_cases = [
        ("Hi there!", "greeting"),
        ("What is offline transaction?", "domain question"),
        ("How's the weather today?", "out of domain"),
        ("How to check my wallet balance?", "domain question"), 
        ("Tell me a joke", "out of domain"),
        ("Thanks for your help!", "thanks"),
        ("Bye!", "farewell")
    ]
    
    for message, expected_type in test_cases:
        print(f"\n👤 User: {message}")
        print(f"   🎯 Expected: {expected_type}")
        
        try:
            response = requests.post(f"{base_url}/chat", json={
                "user_id": "test_user_002", 
                "message": message
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"🤖 Wall-E: {data['response']}")
                print(f"   ✅ Actual: {'Domain' if data['is_domain_related'] else 'General'} | Response Type: {data.get('response_type', 'N/A')}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            break
        
        time.sleep(1)

def test_greeting_variations():
    base_url = "http://localhost:8000"
    
    print("\n👋 Testing Greeting Variations...")
    print("=" * 60)
    
    greetings = [
        "Hello",
        "Hi",
        "Hey",
        "Good morning", 
        "Good afternoon",
        "Good evening",
        "What's up?",
        "Yo"
    ]
    
    for greeting in greetings:
        print(f"\n👤 User: {greeting}")
        
        try:
            response = requests.post(f"{base_url}/chat", json={
                "user_id": "test_user_003",
                "message": greeting
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"🤖 Wall-E: {data['response']}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            break
        
        time.sleep(0.5)

def test_farewell_variations():
    base_url = "http://localhost:8000"
    
    print("\n👋 Testing Farewell Variations...")
    print("=" * 60)
    
    farewells = [
        "Bye",
        "Goodbye", 
        "See you",
        "See ya",
        "Take care",
        "Farewell",
        "Cya",
        "Bye bye"
    ]
    
    for farewell in farewells:
        print(f"\n👤 User: {farewell}")
        
        try:
            response = requests.post(f"{base_url}/chat", json={
                "user_id": "test_user_004",
                "message": farewell
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"🤖 Wall-E: {data['response']}")
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            break
        
        time.sleep(0.5)

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Conversation Tests...")
    print("Note: Make sure the chatbot server is running on http://localhost:8000")
    print()
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    test_greeting_variations()
    test_complete_conversation() 
    test_mixed_conversation()
    test_farewell_variations()
    
    print("\n✅ All conversation tests completed!")