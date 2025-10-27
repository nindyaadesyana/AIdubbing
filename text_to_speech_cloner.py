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
from gtts import gTTS
import io

class TextToSpeechCloner:
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
            
            print(f"Training TTS voice model...")
            
            # Training simulation
            for i in range(10):
                time.sleep(0.5)
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
                print(f"Training progress: {progress}%")
            
            # Analyze voice characteristics
            voice_profile = self._analyze_voice_characteristics(dataset_path, audio_files)
            
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
            
            print(f"TTS voice model trained successfully!")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _analyze_voice_characteristics(self, dataset_path, audio_files):
        """Analyze voice characteristics for TTS conversion"""
        
        pitch_values = []
        spectral_features = []
        
        for audio_file in audio_files[:5]:  # Analyze top 5 samples
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) > sr * 0.3:
                    # Pitch analysis
                    pitches, magnitudes = librosa.piptrack(y=y_clean, sr=sr)
                    pitch_contour = []
                    
                    for t in range(pitches.shape[1]):
                        index = magnitudes[:, t].argmax()
                        pitch = pitches[index, t]
                        if 80 < pitch < 400:
                            pitch_contour.append(pitch)
                    
                    if len(pitch_contour) > 5:
                        pitch_values.extend(pitch_contour)
                        
                        # Spectral features
                        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_clean, sr=sr))
                        spectral_features.append(spectral_centroid)
                        
            except Exception as e:
                continue
        
        if not pitch_values:
            raise Exception("Could not analyze voice characteristics")
        
        return {
            'voice_id': dataset_path.split('/')[-3],
            'pitch_mean': float(np.mean(pitch_values)),
            'pitch_std': float(np.std(pitch_values)),
            'spectral_mean': float(np.mean(spectral_features)) if spectral_features else 2000.0,
            'sample_count': len(audio_files),
            'created_at': datetime.now().isoformat()
        }
    
    def generate_speech(self, voice_id, text, output_path):
        """Generate speech from text using voice characteristics"""
        
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model '{voice_id}' not found")
        
        try:
            print(f"Generating TTS for: {text}")
            
            # Step 1: Generate base TTS using gTTS
            tts = gTTS(text=text, lang='id', slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                base_tts_path = tmp_file.name
            
            # Step 2: Convert to WAV and load
            y_base, sr = librosa.load(base_tts_path, sr=22050)
            os.unlink(base_tts_path)  # Clean up temp file
            
            # Step 3: Apply voice characteristics
            profile = self.voice_models[voice_id]
            target_pitch = profile['pitch_mean']
            
            # Calculate pitch shift needed
            # Get current pitch of TTS
            pitches, magnitudes = librosa.piptrack(y=y_base, sr=sr)
            current_pitches = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if 80 < pitch < 400:
                    current_pitches.append(pitch)
            
            if current_pitches:
                current_pitch = np.mean(current_pitches)
                
                # Calculate semitone shift
                semitone_shift = 12 * np.log2(target_pitch / current_pitch)
                
                # Limit shift to reasonable range
                semitone_shift = np.clip(semitone_shift, -12, 12)
                
                print(f"Applying pitch shift: {semitone_shift:.1f} semitones")
                
                # Apply pitch shift
                y_shifted = librosa.effects.pitch_shift(y_base, sr=sr, n_steps=semitone_shift)
            else:
                y_shifted = y_base
            
            # Step 4: Apply spectral shaping (subtle)
            # Apply gentle formant shifting to match voice character
            spectral_target = profile.get('spectral_mean', 2000.0)
            
            # Simple spectral envelope adjustment
            stft = librosa.stft(y_shifted)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply gentle spectral tilt
            freq_bins = magnitude.shape[0]
            spectral_adjustment = np.linspace(0.9, 1.1, freq_bins).reshape(-1, 1)
            
            if spectral_target > 2500:  # Brighter voice
                spectral_adjustment = np.linspace(0.8, 1.2, freq_bins).reshape(-1, 1)
            elif spectral_target < 1500:  # Darker voice
                spectral_adjustment = np.linspace(1.2, 0.8, freq_bins).reshape(-1, 1)
            
            magnitude_adjusted = magnitude * spectral_adjustment
            
            # Reconstruct audio
            stft_adjusted = magnitude_adjusted * np.exp(1j * phase)
            y_final = librosa.istft(stft_adjusted)
            
            # Step 5: Normalize and save
            y_final = y_final / np.max(np.abs(y_final)) * 0.8
            
            sf.write(output_path, y_final, sr)
            
            print(f"TTS speech generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating TTS speech: {e}")
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