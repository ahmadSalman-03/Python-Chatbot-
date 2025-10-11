import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import os
import re
import random
from .database import database

class GenerativeChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.is_trained = False
        
        # Conversation context tracking
        self.user_sessions = {}
        
        self.load_model()
        self.setup_generator()
        self.setup_conversation_patterns()
    
    def setup_conversation_patterns(self):
        """Setup greeting and farewell patterns"""
        self.greetings = [
            "Hello! I'm Wall-E, your digital payment assistant. How can I help you with transactions today?",
            "Hi there! I'm Wall-E, ready to assist you with payments, security, and wallet services. What do you need help with?",
            "Welcome! I'm Wall-E, your payment companion. How can I make your financial transactions easier today?",
            "Greetings! I'm Wall-E, here to help with all your payment needs. What would you like to know about our services?"
        ]
        
        self.farewells = [
            "Goodbye! Feel free to reach out if you need help with payments or transactions. Have a great day!",
            "See you later! Remember, I'm always here to help with your Wall-E wallet and payment questions.",
            "Take care! Don't hesitate to ask if you need assistance with transactions or wallet features.",
            "Bye for now! I'm here whenever you need help with payments, security, or rewards."
        ]
        
        self.thanks_responses = [
            "You're welcome! Happy to help with your payment needs.",
            "Glad I could assist! Let me know if you need anything else with Wall-E.",
            "Anytime! I'm here whenever you have questions about transactions or wallet features.",
            "My pleasure! Feel free to ask if you need more help with payments."
        ]
    
    def load_model(self):
        """Load the generative model"""
        try:
            print("🔄 Loading generative model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            print("✅ Generative model loaded successfully!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.model = None
    
    def setup_generator(self):
        """Setup text generation pipeline"""
        if self.model:
            try:
                self.generator = pipeline(
                    'text-generation',
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1  # Use CPU
                )
                print("✅ Text generator ready!")
            except Exception as e:
                print(f"❌ Generator setup failed: {e}")
                self.generator = None
    
    def train_from_file(self, file_path):
        """Train the model from training file"""
        if not self.model:
            print("❌ Model not available for training")
            return False
        
        try:
            print(f"📚 Training from {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse training data
            training_pairs = self.parse_training_data(content)
            
            # Store in database with embeddings
            for question, answer in training_pairs:
                database.store_training_embedding(question, answer)
            
            self.is_trained = True
            print(f"✅ Trained on {len(training_pairs)} question-answer pairs!")
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def parse_training_data(self, content):
        """Parse training data from text file"""
        pairs = []
        
        # Split by USER: and CHATBOT: pattern
        pattern = r'USER:\s*(.*?)\s*CHATBOT:\s*(.*?)(?=USER:|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for user_msg, bot_resp in matches:
            user_msg = user_msg.strip()
            bot_resp = bot_resp.strip()
            
            if user_msg and bot_resp:
                pairs.append((user_msg, bot_resp))
        
        return pairs
    
    def is_greeting(self, text):
        """Check if message is a greeting"""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'hola']
        return any(greet in text.lower() for greet in greetings)
    
    def is_farewell(self, text):
        """Check if message is a farewell"""
        farewells = ['bye', 'goodbye', 'see you', 'see ya', 'farewell', 'cya', 'take care']
        return any(farewell in text.lower() for farewell in farewells)
    
    def is_thanks(self, text):
        """Check if message is expressing thanks"""
        thanks = ['thank', 'thanks', 'thank you', 'appreciate', 'grateful']
        return any(thx in text.lower() for thx in thanks)
    
    def get_user_session(self, user_id):
        """Get or create user session"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'message_count': 0,
                'last_message_time': None,
                'conversation_started': False
            }
        return self.user_sessions[user_id]
    
    def generate_response(self, user_message, user_id="default_user"):
        """Generate response using generative AI with conversation handling"""
        # Get user session
        session = self.get_user_session(user_id)
        session['message_count'] += 1
        
        # Handle greetings
        if self.is_greeting(user_message) and not session['conversation_started']:
            session['conversation_started'] = True
            return random.choice(self.greetings), True
        
        # Handle farewells
        if self.is_farewell(user_message):
            session['conversation_started'] = False
            session['message_count'] = 0
            return random.choice(self.farewells), True
        
        # Handle thanks
        if self.is_thanks(user_message):
            return random.choice(self.thanks_responses), True
        
        # First check if within domain
        is_domain_related = database.is_within_domain(user_message)
        
        if not is_domain_related:
            return self.get_out_of_domain_response(), False
        
        # Try semantic search first
        similar_answer = database.semantic_search(user_message)
        if similar_answer:
            return similar_answer, True
        
        # Generate new response using AI
        if self.generator:
            try:
                # Create context-aware prompt
                prompt = self.create_prompt(user_message)
                
                # Generate response
                generated_text = self.generator(
                    prompt,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )[0]['generated_text']
                
                # Extract the response part
                response = self.extract_response(generated_text, prompt)
                
                if self.is_valid_response(response):
                    # Store this new knowledge
                    database.store_training_embedding(user_message, response)
                    return response, True
                
            except Exception as e:
                print(f"❌ Generation error: {e}")
        
        # Fallback to domain-appropriate response
        return self.get_domain_fallback_response(user_message), True
    
    def create_prompt(self, user_message):
        """Create context-aware prompt for generation"""
        base_context = """
        You are Wall-E, a helpful AI assistant for a digital wallet and payment application. 
        You specialize in helping users with financial transactions, wallet management, 
        security features, rewards, and payment-related queries.
        
        Keep responses professional, helpful, and focused on the Wall-E payment platform.
        Provide clear, step-by-step instructions when explaining processes.
        Be conversational but stay on topic about payment services.
        
        User: {user_message}
        Wall-E:"""
        
        return base_context.format(user_message=user_message)
    
    def extract_response(self, generated_text, prompt):
        """Extract the response from generated text"""
        if prompt in generated_text:
            response = generated_text.split(prompt)[-1].strip()
        else:
            response = generated_text.strip()
        
        # Clean up response
        response = re.split(r'[\n]+', response)[0]  # Take first paragraph
        response = response.replace('"', '').replace("'", "")
        
        return response
    
    def is_valid_response(self, response):
        """Validate if generated response is appropriate"""
        if not response or len(response) < 10:
            return False
        
        invalid_indicators = [
            'sorry, i cannot', 'i am not able to', 'as an ai', 'i don\'t have',
            '...', 'i cannot answer', 'i don\'t know how'
        ]
        
        response_lower = response.lower()
        return not any(indicator in response_lower for indicator in invalid_indicators)
    
    def get_out_of_domain_response(self):
        """Response for out-of-domain queries"""
        return "I'm specialized in helping with Wall-E payment services and cannot answer general questions. Would you like to know about sending money, wallet security, or making offline transactions?"
    
    def get_domain_fallback_response(self, user_message):
        """Fallback response for domain questions when generation fails"""
        domain_fallbacks = {
            'send': "To send money, go to the Send section, enter recipient details, specify amount, and authenticate. Would you like me to explain online or offline transfer methods?",
            'balance': "Your wallet balance is visible on the main dashboard. For transaction history, check the History section with detailed records.",
            'secure': "Wall-E uses multiple security layers: encryption, biometric auth, and real-time monitoring. Your transactions are protected 24/7.",
            'offline': "Offline payments use QR codes - scan, enter amount, confirm. No internet needed! Transactions sync when connected.",
            'reward': "Earn coins on transactions! Redeem for discounts, top-ups, and bill payments. Check Rewards section for current offers."
        }
        
        user_lower = user_message.lower()
        for keyword, response in domain_fallbacks.items():
            if keyword in user_lower:
                return response
        
        return "I can help you with Wall-E payment services, wallet management, security features, and transaction support. What specific assistance do you need?"

# Global chatbot instance
chatbot = GenerativeChatbot()