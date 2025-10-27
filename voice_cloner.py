import os
import json
import uuid
import threading
from datetime import datetime
# TTS imports will be loaded when needed
# from TTS.api import TTS

class VoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.tts_models = {}
        
        # Buat direktori jika belum ada
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load existing models
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load model yang sudah ada"""
        for voice_dir in os.listdir(self.models_dir):
            voice_path = os.path.join(self.models_dir, voice_dir)
            if os.path.isdir(voice_path):
                model_path = os.path.join(voice_path, "best_model.pth")
                config_path = os.path.join(voice_path, "config.json")
                
                if os.path.exists(model_path) and os.path.exists(config_path):
                    try:
                        tts = TTS(model_path=model_path, config_path=config_path)
                        self.tts_models[voice_dir] = tts
                    except Exception as e:
                        print(f"Error loading model {voice_dir}: {e}")
    
    def start_training(self, voice_id, epochs=100):
        """Mulai training voice model"""
        training_id = str(uuid.uuid4())
        
        self.training_status[training_id] = {
            'status': 'starting',
            'progress': 0,
            'voice_id': voice_id,
            'epochs': epochs,
            'start_time': datetime.now().isoformat()
        }
        
        # Jalankan training di thread terpisah
        thread = threading.Thread(
            target=self._train_model,
            args=(training_id, voice_id, epochs)
        )
        thread.daemon = True
        thread.start()
        
        return training_id
    
    def _train_model(self, training_id, voice_id, epochs):
        """Training model (dijalankan di background)"""
        try:
            # Simulate training for demo
            import time
            self.training_status[training_id]['status'] = 'training'
            
            for i in range(10):
                time.sleep(2)  # Simulate training time
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
            
            # Mark as completed
            self.training_status[training_id].update({
                'status': 'completed',
                'progress': 100,
                'end_time': datetime.now().isoformat()
            })
            
            # Add to available models (mock)
            self.tts_models[voice_id] = f"mock_model_{voice_id}"
                
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def get_training_status(self, training_id):
        """Get status training"""
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def generate_speech(self, text, voice_id):
        """Generate speech dari text"""
        if voice_id not in self.tts_models:
            raise Exception(f"Voice model {voice_id} not found or not trained")
        
        # Mock generation for demo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{voice_id}_{timestamp}.wav"
        output_path = os.path.join("outputs", output_filename)
        
        os.makedirs("outputs", exist_ok=True)
        
        # Create a simple mock audio file
        import numpy as np
        import soundfile as sf
        
        # Generate 2 seconds of sine wave as demo
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
        
        sf.write(output_path, audio, sample_rate)
        
        return output_path
    
    def list_available_voices(self):
        """List semua voice yang tersedia"""
        voices = []
        for voice_id in self.tts_models.keys():
            voices.append({
                'id': voice_id,
                'name': voice_id.replace('_', ' ').title(),
                'status': 'ready'
            })
        
        # Tambahkan voice yang sedang training
        for training_id, status in self.training_status.items():
            if status['status'] in ['starting', 'training']:
                voice_id = status['voice_id']
                if voice_id not in [v['id'] for v in voices]:
                    voices.append({
                        'id': voice_id,
                        'name': voice_id.replace('_', ' ').title(),
                        'status': 'training',
                        'progress': status.get('progress', 0)
                    })
        
        return voices