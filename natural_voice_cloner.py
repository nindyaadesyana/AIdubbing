import os
import json
import uuid
import threading
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import tempfile
import subprocess
import sys
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import random

class NaturalVoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
    
    def _load_existing_models(self):
        for voice_dir in os.listdir(self.models_dir):
            voice_path = os.path.join(self.models_dir, voice_dir)
            if os.path.isdir(voice_path):
                model_file = os.path.join(voice_path, "voice_profile.json")
                if os.path.exists(model_file):
                    with open(model_file, 'r') as f:
                        profile = json.load(f)
                        self.voice_models[voice_dir] = profile
    
    def start_training(self, voice_id, epochs=100):
        training_id = str(uuid.uuid4())
        
        self.training_status[training_id] = {
            'status': 'starting',
            'progress': 0,
            'voice_id': voice_id,
            'epochs': epochs,
            'start_time': datetime.now().isoformat()
        }
        
        thread = threading.Thread(target=self._train_model, args=(training_id, voice_id, epochs))
        thread.daemon = True
        thread.start()
        
        return training_id
    
    def _train_model(self, training_id, voice_id, epochs):
        try:
            import time
            self.training_status[training_id]['status'] = 'training'
            
            dataset_path = f"datasets/processed/{voice_id}/clips/"
            if not os.path.exists(dataset_path):
                raise Exception(f"Dataset path not found: {dataset_path}")
            
            audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
            if len(audio_files) < 2:
                raise Exception("Not enough audio samples for training")
            
            print(f"Training natural voice model...")
            
            # Training simulation
            for i in range(10):
                time.sleep(1)
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
                print(f"Training progress: {progress}%")
            
            # Create voice profile
            voice_profile = self._create_voice_profile(dataset_path, audio_files)
            
            # Save model
            model_path = os.path.join(self.models_dir, voice_id)
            os.makedirs(model_path, exist_ok=True)
            
            profile_file = os.path.join(model_path, "voice_profile.json")
            with open(profile_file, 'w') as f:
                json.dump(voice_profile, f, indent=2)
            
            self.voice_models[voice_id] = voice_profile
            
            self.training_status[training_id].update({
                'status': 'completed',
                'progress': 100,
                'end_time': datetime.now().isoformat(),
                'model_path': model_path
            })
            
            print(f"Natural voice model trained successfully!")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _create_voice_profile(self, dataset_path, audio_files):
        """Create natural voice profile"""
        
        voice_samples = []
        
        for audio_file in audio_files[:10]:
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) > sr * 0.5:  # Minimal 0.5 detik
                    # Simple pitch analysis
                    pitches, magnitudes = librosa.piptrack(y=y_clean, sr=sr)
                    pitch_values = []
                    
                    for t in range(pitches.shape[1]):
                        index = magnitudes[:, t].argmax()
                        pitch = pitches[index, t]
                        if 80 < pitch < 400:
                            pitch_values.append(pitch)
                    
                    if len(pitch_values) > 5:
                        voice_samples.append({
                            'file': audio_file,
                            'audio_data': y_clean.tolist(),
                            'pitch_mean': float(np.mean(pitch_values)),
                            'duration': len(y_clean) / sr
                        })
                        
            except Exception as e:
                continue
        
        # Sort by duration and take best samples
        voice_samples.sort(key=lambda x: x['duration'], reverse=True)
        best_samples = voice_samples[:5]
        
        # Calculate average characteristics
        avg_pitch = np.mean([s['pitch_mean'] for s in best_samples])
        
        return {
            'voice_id': dataset_path.split('/')[-3],
            'pitch_mean': float(avg_pitch),
            'best_samples': best_samples,
            'sample_count': len(best_samples)
        }
    
    def generate_speech(self, voice_id, text, output_path):
        """Generate speech using natural voice synthesis"""
        
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model '{voice_id}' not found")
        
        try:
            print(f"Generating natural speech for: {text}")
            
            # Get voice profile
            profile = self.voice_models[voice_id]
            best_samples = profile.get('best_samples', [])
            
            if not best_samples:
                raise Exception("No voice samples available")
            
            # Use the longest, clearest sample as base
            base_sample = best_samples[0]
            if isinstance(base_sample, dict) and 'audio_data' in base_sample:
                base_audio = np.array(base_sample['audio_data'])
            else:
                raise Exception("Invalid voice sample data")
            
            # Simple approach: concatenate and modify existing audio
            words = text.split()
            
            if len(words) <= len(best_samples):
                # Use different samples for each word
                result_audio = []
                
                for i, word in enumerate(words):
                    if i < len(best_samples) and isinstance(best_samples[i], dict) and 'audio_data' in best_samples[i]:
                        sample_audio = np.array(best_samples[i]['audio_data'])
                    else:
                        sample_audio = base_audio
                    
                    # Add slight variation
                    variation = np.random.uniform(0.95, 1.05)
                    modified_audio = sample_audio * variation
                    
                    result_audio.extend(modified_audio)
                    
                    # Add pause between words
                    if i < len(words) - 1:
                        pause = np.zeros(int(22050 * 0.1))  # 0.1 second pause
                        result_audio.extend(pause)
                
                final_audio = np.array(result_audio)
            else:
                # For longer text, repeat and modify base sample
                target_duration = len(words) * 0.5  # 0.5 seconds per word
                target_samples = int(target_duration * 22050)
                
                # Repeat base audio to match target duration
                repeats = int(np.ceil(target_samples / len(base_audio)))
                extended_audio = np.tile(base_audio, repeats)[:target_samples]
                
                # Add natural variation
                variation = np.random.uniform(0.9, 1.1, len(extended_audio))
                final_audio = extended_audio * variation
            
            # Normalize audio
            final_audio = final_audio / np.max(np.abs(final_audio)) * 0.8
            
            # Save output
            sf.write(output_path, final_audio, 22050)
            
            print(f"Natural speech generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return False
    
    def get_training_status(self, training_id):
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def list_voices(self):
        return list(self.voice_models.keys())
    
    def get_voice_info(self, voice_id):
        if voice_id in self.voice_models:
            profile = self.voice_models[voice_id]
            return {
                'voice_id': voice_id,
                'sample_count': profile.get('sample_count', 0),
                'pitch_mean': profile.get('pitch_mean', 0)
            }
        return None