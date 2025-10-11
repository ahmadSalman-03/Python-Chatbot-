import os
from .chatbot import chatbot

class TrainingController:
    def __init__(self):
        self.training_file = "training_data/training_dataset.txt"
        self.is_initial_training_done = False
    
    def run_initial_training(self):
        """Run initial training from file"""
        print("🚀 Starting initial training phase...")
        
        if not os.path.exists(self.training_file):
            print(f"❌ Training file not found: {self.training_file}")
            return False
        
        success = chatbot.train_from_file(self.training_file)
        
        if success:
            self.is_initial_training_done = True
            print("✅ Initial training completed! Switching to generative mode.")
            return True
        else:
            print("❌ Initial training failed!")
            return False
    
    def get_status(self):
        """Get training status"""
        return {
            'initial_training_done': self.is_initial_training_done,
            'model_loaded': chatbot.model is not None,
            'training_file': self.training_file
        }

# Global trainer instance
trainer = TrainingController()