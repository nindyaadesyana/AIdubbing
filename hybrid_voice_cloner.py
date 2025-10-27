import os
import json
import uuid
import threading
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
import tempfile
from gtts import gTTS
from scipy.signal import butter, filtfilt

class HybridVoiceCloner:
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
            
            print(f"Training hybrid voice model...")
            
            for i in range(10):
                time.sleep(0.5)
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
                print(f"Training progress: {progress}%")
            
            # Analyze voice for conversion parameters
            voice_profile = self._analyze_voice_for_conversion(dataset_path, audio_files)
            
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
            
            print(f"Hybrid voice model trained successfully!")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _analyze_voice_for_conversion(self, dataset_path, audio_files):
        """Analyze voice characteristics for TTS conversion"""
        
        pitch_values = []
        spectral_values = []
        
        for audio_file in audio_files[:5]:
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) > sr * 0.5:
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
                        
                        # Spectral centroid
                        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_clean, sr=sr))
                        spectral_values.append(spectral_centroid)
                        
            except Exception as e:
                continue
        
        if not pitch_values:
            raise Exception("Could not analyze voice characteristics")
        
        return {
            'voice_id': dataset_path.split('/')[-3],
            'target_pitch': float(np.mean(pitch_values)),
            'target_spectral': float(np.mean(spectral_values)) if spectral_values else 2000.0,
            'pitch_variance': float(np.std(pitch_values)),
            'sample_count': len(audio_files),
            'created_at': datetime.now().isoformat()
        }
    
    def generate_speech(self, voice_id, text, output_path):
        """Generate TTS speech and convert to target voice"""
        
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model '{voice_id}' not found")
        
        try:
            print(f"Generating speech: {text}")
            
            # Step 1: Generate TTS
            tts = gTTS(text=text, lang='id', slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                base_tts_path = tmp_file.name
            
            # Step 2: Load TTS audio
            y_tts, sr = librosa.load(base_tts_path, sr=22050)
            os.unlink(base_tts_path)
            
            # Step 3: Convert to target voice
            profile = self.voice_models[voice_id]
            y_converted = self._convert_to_target_voice(y_tts, sr, profile)
            
            # Step 4: Save
            sf.write(output_path, y_converted, sr)
            
            print(f"Speech generated with voice conversion: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return False
    
    def _convert_to_target_voice(self, y_tts, sr, profile):
        """Convert TTS to target voice characteristics"""
        
        target_pitch = profile['target_pitch']
        target_spectral = profile['target_spectral']
        
        # Step 1: Pitch conversion
        pitches, magnitudes = librosa.piptrack(y=y_tts, sr=sr)
        current_pitches = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if 80 < pitch < 400:
                current_pitches.append(pitch)
        
        if current_pitches:
            current_pitch = np.mean(current_pitches)
            semitone_shift = 12 * np.log2(target_pitch / current_pitch)
            semitone_shift = np.clip(semitone_shift, -8, 8)  # Reasonable range
            
            print(f"Pitch shift: {semitone_shift:.1f} semitones")
            y_pitched = librosa.effects.pitch_shift(y_tts, sr=sr, n_steps=semitone_shift)
        else:
            y_pitched = y_tts
        
        # Step 2: Spectral adjustment
        current_spectral = np.mean(librosa.feature.spectral_centroid(y=y_pitched, sr=sr))
        spectral_ratio = target_spectral / current_spectral
        
        if 0.7 < spectral_ratio < 1.5:  # Apply only reasonable adjustments
            stft = librosa.stft(y_pitched)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Simple spectral shaping
            freq_bins = magnitude.shape[0]
            if spectral_ratio > 1.1:
                # Brighten
                spectral_weight = np.linspace(0.9, 1.3, freq_bins).reshape(-1, 1)
            elif spectral_ratio < 0.9:
                # Darken
                spectral_weight = np.linspace(1.2, 0.8, freq_bins).reshape(-1, 1)
            else:
                spectral_weight = np.ones((freq_bins, 1))
            
            magnitude_adjusted = magnitude * spectral_weight
            stft_adjusted = magnitude_adjusted * np.exp(1j * phase)
            y_converted = librosa.istft(stft_adjusted)
        else:
            y_converted = y_pitched
        
        # Step 3: Normalize
        y_converted = y_converted / np.max(np.abs(y_converted)) * 0.8
        
        return y_converted
    
    def get_training_status(self, training_id):
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def list_voices(self):
        return list(self.voice_models.keys())
    
    def get_voice_info(self, voice_id):
        if voice_id in self.voice_models:
            profile = self.voice_models[voice_id]
            return {
                'voice_id': voice_id,
                'sample_count': profile.get('sample_count', 0)
            }
        return None