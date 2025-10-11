# from pymongo import MongoClient
# from datetime import datetime
# import uuid
# import os
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from dotenv import load_dotenv

# load_dotenv()

# # Download NLTK data
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     nltk.download('wordnet', quiet=True)
# except:
#     pass

# class NLPDatabase:
#     def __init__(self):
#         self.client = None
#         self.db = None
#         self.nlp = None
#         self.sentence_model = None
#         self.domain_keywords = set()
        
#         self.load_nlp_models()
#         self.connect()
#         self.setup_domain_keywords()
    
#     def load_nlp_models(self):
#         """Load NLP models for text processing"""
#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#             print("✅ spaCy model loaded")
#         except:
#             print("❌ spaCy model not found, install with: python -m spacy download en_core_web_sm")
#             self.nlp = None
        
#         try:
#             self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
#             print("✅ Sentence transformer loaded")
#         except Exception as e:
#             print(f"❌ Sentence transformer failed: {e}")
#             self.sentence_model = None
    
#     def setup_domain_keywords(self):
#         """Setup domain-specific keywords for filtering"""
#         self.domain_keywords = {
#             # Payment related
#             'send', 'transfer', 'money', 'payment', 'pay', 'transaction', 'fund', 'amount',
#             # Wallet features
#             'wallet', 'balance', 'account', 'recharge', 'topup', 'add money',
#             # Security
#             'secure', 'safe', 'protection', 'encryption', 'pin', 'biometric', 'authentication',
#             # Offline features
#             'offline', 'qr', 'code', 'scan', 'internet', 'connectivity', 'network',
#             # Rewards
#             'reward', 'point', 'coin', 'discount', 'voucher', 'bonus', 'loyalty',
#             # Support
#             'help', 'support', 'contact', 'customer', 'issue', 'problem',
#             # Banking terms
#             'bank', 'account', 'transfer', 'withdraw', 'deposit', 'limit', 'fee',
#             # App specific
#             'wall-e', 'app', 'application', 'mobile', 'phone'
#         }
    
#     def connect(self):
#         try:
#             self.client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017'))
#             self.db = self.client['wall_e_generative_chatbot']
            
#             self.conversations = self.db['conversations']
#             self.training_embeddings = self.db['training_embeddings']
#             self.knowledge_base = self.db['knowledge_base']
            
#             print("✅ Connected to MongoDB with NLP capabilities")
#         except Exception as e:
#             print(f"❌ MongoDB connection failed: {e}")
    
#     def is_within_domain(self, text):
#         """Check if text is within Wall-E domain using NLP"""
#         if not self.nlp:
#             return self.basic_domain_check(text)
        
#         try:
#             doc = self.nlp(text.lower())
            
#             # Extract nouns and verbs
#             relevant_words = []
#             for token in doc:
#                 if (token.pos_ in ['NOUN', 'VERB', 'PROPN'] and 
#                     not token.is_stop and len(token.text) > 2):
#                     relevant_words.append(token.lemma_)
            
#             # Check domain relevance
#             domain_matches = len(set(relevant_words) & self.domain_keywords)
#             total_relevant = len(relevant_words)
            
#             if total_relevant == 0:
#                 return False
            
#             domain_ratio = domain_matches / total_relevant
#             return domain_ratio >= 0.3  # At least 30% domain relevance
            
#         except Exception as e:
#             print(f"❌ Domain check error: {e}")
#             return self.basic_domain_check(text)
    
#     def basic_domain_check(self, text):
#         """Fallback domain check using keywords"""
#         text_lower = text.lower()
#         domain_terms_found = sum(1 for keyword in self.domain_keywords if keyword in text_lower)
#         return domain_terms_found >= 2
    
#     def store_training_embedding(self, question, answer):
#         """Store question-answer pairs with embeddings for semantic search"""
#         if not self.sentence_model:
#             return
        
#         try:
#             # Create embedding for semantic search
#             question_embedding = self.sentence_model.encode([question])[0]
            
#             training_item = {
#                 'pair_id': str(uuid.uuid4()),
#                 'question': question,
#                 'answer': answer,
#                 'question_embedding': question_embedding.tolist(),
#                 'domain_score': self.calculate_domain_relevance(question),
#                 'created_at': datetime.utcnow(),
#                 'usage_count': 0
#             }
            
#             self.training_embeddings.insert_one(training_item)
#             print(f"💾 Stored training embedding: {question[:50]}...")
            
#         except Exception as e:
#             print(f"❌ Error storing training embedding: {e}")
    
#     def semantic_search(self, user_question, threshold=0.7):
#         """Find similar questions using semantic search"""
#         if not self.sentence_model:
#             return None
        
#         try:
#             user_embedding = self.sentence_model.encode([user_question])[0]
            
#             # Get all training embeddings
#             all_training = list(self.training_embeddings.find({}))
            
#             if not all_training:
#                 return None
            
#             # Calculate similarities
#             best_match = None
#             best_similarity = 0
            
#             for item in all_training:
#                 stored_embedding = np.array(item['question_embedding'])
#                 similarity = cosine_similarity([user_embedding], [stored_embedding])[0][0]
                
#                 if similarity > best_similarity and similarity >= threshold:
#                     best_similarity = similarity
#                     best_match = item
            
#             if best_match:
#                 # Update usage count
#                 self.training_embeddings.update_one(
#                     {'pair_id': best_match['pair_id']},
#                     {'$inc': {'usage_count': 1}}
#                 )
#                 return best_match['answer']
            
#         except Exception as e:
#             print(f"❌ Semantic search error: {e}")
        
#         return None
    
#     def calculate_domain_relevance(self, text):
#         """Calculate how relevant text is to Wall-E domain"""
#         if not self.nlp:
#             return 0.5
        
#         try:
#             doc = self.nlp(text.lower())
#             relevant_terms = [token.lemma_ for token in doc 
#                             if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop]
            
#             if not relevant_terms:
#                 return 0.0
            
#             domain_matches = len(set(relevant_terms) & self.domain_keywords)
#             return domain_matches / len(relevant_terms)
            
#         except:
#             return 0.5
    
#     def store_conversation(self, user_id, user_message, bot_response, is_domain_related):
#         """Store conversation with NLP analysis"""
#         conversation = {
#             'conversation_id': str(uuid.uuid4()),
#             'user_id': user_id,
#             'user_message': user_message,
#             'bot_response': bot_response,
#             'is_domain_related': is_domain_related,
#             'domain_score': self.calculate_domain_relevance(user_message),
#             'timestamp': datetime.utcnow(),
#             'used_for_training': False
#         }
        
#         self.conversations.insert_one(conversation)
#         return conversation['conversation_id']
    
#     def get_domain_conversations(self, min_domain_score=0.4):
#         """Get domain-relevant conversations for training"""
#         return list(self.conversations.find({
#             'domain_score': {'$gte': min_domain_score},
#             'used_for_training': False
#         }))

# # Global database instance
# database = NLPDatabase()


from datetime import datetime
import uuid
import os
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class NLPDatabase:
    def __init__(self):
        # In-memory storage only (no MongoDB)
        self.conversations = []
        self.training_embeddings = []
        self.knowledge_base = []
        
        self.nlp = None
        self.sentence_model = None
        self.domain_keywords = set()
        
        self.load_nlp_models()
        self.setup_domain_keywords()
        print("✅ Database running in MEMORY-ONLY mode (no MongoDB)")
    
    def load_nlp_models(self):
        """Load NLP models for text processing"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        except:
            print("❌ spaCy model not found, using basic NLP")
            self.nlp = None
        
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded")
        except Exception as e:
            print(f"❌ Sentence transformer failed: {e}")
            self.sentence_model = None
    
    def setup_domain_keywords(self):
        """Setup domain-specific keywords for filtering"""
        self.domain_keywords = {
            # Payment related
            'send', 'transfer', 'money', 'payment', 'pay', 'transaction', 'fund', 'amount',
            # Wallet features
            'wallet', 'balance', 'account', 'recharge', 'topup', 'add money',
            # Security
            'secure', 'safe', 'protection', 'encryption', 'pin', 'biometric', 'authentication',
            # Offline features
            'offline', 'qr', 'code', 'scan', 'internet', 'connectivity', 'network',
            # Rewards
            'reward', 'point', 'coin', 'discount', 'voucher', 'bonus', 'loyalty',
            # Support
            'help', 'support', 'contact', 'customer', 'issue', 'problem',
            # Banking terms
            'bank', 'account', 'transfer', 'withdraw', 'deposit', 'limit', 'fee',
            # App specific
            'wall-e', 'app', 'application', 'mobile', 'phone'
        }
    
    def is_within_domain(self, text):
        """Check if text is within Wall-E domain using NLP"""
        if not self.nlp:
            return self.basic_domain_check(text)
        
        try:
            doc = self.nlp(text.lower())
            
            # Extract nouns and verbs
            relevant_words = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'VERB', 'PROPN'] and 
                    not token.is_stop and len(token.text) > 2):
                    relevant_words.append(token.lemma_)
            
            # Check domain relevance
            domain_matches = len(set(relevant_words) & self.domain_keywords)
            total_relevant = len(relevant_words)
            
            if total_relevant == 0:
                return False
            
            domain_ratio = domain_matches / total_relevant
            return domain_ratio >= 0.3  # At least 30% domain relevance
            
        except Exception as e:
            print(f"❌ Domain check error: {e}")
            return self.basic_domain_check(text)
    
    def basic_domain_check(self, text):
        """Fallback domain check using keywords"""
        text_lower = text.lower()
        domain_terms_found = sum(1 for keyword in self.domain_keywords if keyword in text_lower)
        return domain_terms_found >= 2
    
    def store_training_embedding(self, question, answer):
        """Store question-answer pairs with embeddings for semantic search"""
        if not self.sentence_model:
            return
        
        try:
            # Create embedding for semantic search
            question_embedding = self.sentence_model.encode([question])[0]
            
            training_item = {
                'pair_id': str(uuid.uuid4()),
                'question': question,
                'answer': answer,
                'question_embedding': question_embedding.tolist(),
                'domain_score': self.calculate_domain_relevance(question),
                'created_at': datetime.utcnow(),
                'usage_count': 0
            }
            
            self.training_embeddings.append(training_item)
            print(f"💾 Stored training embedding: {question[:50]}...")
            
        except Exception as e:
            print(f"❌ Error storing training embedding: {e}")
    
    def semantic_search(self, user_question, threshold=0.7):
        """Find similar questions using semantic search"""
        if not self.sentence_model or not self.training_embeddings:
            return None
        
        try:
            user_embedding = self.sentence_model.encode([user_question])[0]
            
            # Calculate similarities
            best_match = None
            best_similarity = 0
            
            for item in self.training_embeddings:
                stored_embedding = np.array(item['question_embedding'])
                similarity = cosine_similarity([user_embedding], [stored_embedding])[0][0]
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = item
            
            if best_match:
                # Update usage count
                best_match['usage_count'] += 1
                return best_match['answer']
            
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
        
        return None
    
    def calculate_domain_relevance(self, text):
        """Calculate how relevant text is to Wall-E domain"""
        if not self.nlp:
            return 0.5
        
        try:
            doc = self.nlp(text.lower())
            relevant_terms = [token.lemma_ for token in doc 
                            if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop]
            
            if not relevant_terms:
                return 0.0
            
            domain_matches = len(set(relevant_terms) & self.domain_keywords)
            return domain_matches / len(relevant_terms)
            
        except:
            return 0.5
    
    def store_conversation(self, user_id, user_message, bot_response, is_domain_related):
        """Store conversation with NLP analysis"""
        conversation = {
            'conversation_id': str(uuid.uuid4()),
            'user_id': user_id,
            'user_message': user_message,
            'bot_response': bot_response,
            'is_domain_related': is_domain_related,
            'domain_score': self.calculate_domain_relevance(user_message),
            'timestamp': datetime.utcnow(),
            'used_for_training': False
        }
        
        self.conversations.append(conversation)
        print(f"💾 Stored conversation: {user_message[:30]}...")
        return conversation['conversation_id']
    
    def get_domain_conversations(self, min_domain_score=0.4):
        """Get domain-relevant conversations for training"""
        return [conv for conv in self.conversations if conv['domain_score'] >= min_domain_score]

# Global database instance
database = NLPDatabase()