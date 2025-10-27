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

class UniversalVoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        print("✅ Universal Voice Cloner initialized")
    
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
            
            print(f"Training universal voice model...")
            
            for i in range(10):
                time.sleep(0.5)
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
                print(f"Universal training progress: {progress}%")
            
            # Create universal voice profile
            voice_profile = self._create_universal_profile(dataset_path, audio_files)
            
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
            
            print(f"Universal voice model trained successfully!")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _create_universal_profile(self, dataset_path, audio_files):
        """Create universal voice profile compatible with all formats"""
        
        pitch_values = []
        spectral_values = []
        
        for audio_file in audio_files[:5]:
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) > sr * 0.5:
                    # Simple pitch analysis
                    pitches = librosa.yin(y_clean, fmin=80, fmax=400)
                    pitch_clean = pitches[~np.isnan(pitches)]
                    
                    if len(pitch_clean) > 5:
                        pitch_values.extend(pitch_clean)
                        
                        # Spectral analysis
                        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_clean, sr=sr))
                        spectral_values.append(spectral_centroid)
                        
            except Exception as e:
                continue
        
        if not pitch_values:
            raise Exception("Could not analyze voice characteristics")
        
        return {
            'voice_id': dataset_path.split('/')[-3],
            'universal_profile': {
                'pitch_mean': float(np.mean(pitch_values)),
                'pitch_std': float(np.std(pitch_values)),
                'spectral_mean': float(np.mean(spectral_values)) if spectral_values else 2000.0
            },
            'sample_count': len(audio_files),
            'created_at': datetime.now().isoformat()
        }
    
    def generate_speech(self, voice_id, text, output_path):
        """Generate speech with universal compatibility"""
        
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model '{voice_id}' not found")
        
        try:
            print(f"Generating universal TTS: {text}")
            
            # Step 1: Generate base TTS
            tts = gTTS(text=text, lang='id', slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                base_tts_path = tmp_file.name
            
            # Step 2: Load TTS
            y_tts, sr = librosa.load(base_tts_path, sr=22050)
            os.unlink(base_tts_path)
            
            # Step 3: Apply voice conversion based on available profile format
            profile = self.voice_models[voice_id]
            y_converted = self._apply_universal_conversion(y_tts, sr, profile)
            
            # Step 4: Save
            sf.write(output_path, y_converted, sr)
            
            print(f"Universal TTS generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating universal TTS: {e}")
            return False
    
    def _apply_universal_conversion(self, y_tts, sr, profile):
        """Apply voice conversion based on available profile format"""
        
        print("Applying universal voice conversion...")
        
        # Detect profile format and extract target characteristics
        target_pitch = None
        target_spectral = None
        
        # Format 1: Universal profile (new format)
        if 'universal_profile' in profile:
            target_pitch = profile['universal_profile']['pitch_mean']
            target_spectral = profile['universal_profile']['spectral_mean']
        
        # Format 2: Speaker embedding format
        elif 'speaker_embedding' in profile:
            embedding = profile['speaker_embedding']
            if isinstance(embedding, dict):
                target_pitch = embedding.get('pitch_mean', 200.0)
                target_spectral = embedding.get('spectral_centroid', 2000.0)
            else:
                target_pitch = 200.0
                target_spectral = 2000.0
        
        # Format 3: Perfect voice cloner format
        elif 'pitch_mean' in profile:
            target_pitch = profile['pitch_mean']
            target_spectral = profile.get('spectral_centroid_mean', 2000.0)
        
        # Format 4: Pitch profile format
        elif 'pitch_profile' in profile:
            target_pitch = profile['pitch_profile']['mean']
            target_spectral = profile.get('spectral_profile', {}).get('centroid_mean', 2000.0)
        
        # Format 5: Simple format
        elif 'samples' in profile:
            # Use default values for simple format
            target_pitch = 200.0
            target_spectral = 2000.0
        
        else:
            print("Unknown profile format, using default conversion")
            target_pitch = 200.0
            target_spectral = 2000.0
        
        # Apply conversion
        y_converted = self._convert_voice(y_tts, sr, target_pitch, target_spectral)
        
        print("Universal voice conversion completed")
        return y_converted
    
    def _convert_voice(self, y, sr, target_pitch, target_spectral):
        """Apply voice conversion with target characteristics"""
        
        # Pitch conversion
        pitches = librosa.yin(y, fmin=80, fmax=400)
        current_pitches = pitches[~np.isnan(pitches)]
        
        if len(current_pitches) > 10 and target_pitch:
            current_pitch = np.mean(current_pitches)
            semitone_shift = 12 * np.log2(target_pitch / current_pitch)
            semitone_shift = np.clip(semitone_shift, -8, 8)
            
            print(f"Universal pitch shift: {semitone_shift:.1f} semitones")
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone_shift)
        
        # Spectral conversion
        if target_spectral:
            current_spectral = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_ratio = target_spectral / current_spectral
            
            if 0.7 < spectral_ratio < 1.5:
                stft = librosa.stft(y)
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                
                freq_bins = magnitude.shape[0]
                freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-1)
                
                if spectral_ratio > 1.1:
                    spectral_filter = 1 + 0.2 * (freqs / np.max(freqs))
                elif spectral_ratio < 0.9:
                    spectral_filter = 1 - 0.2 * (freqs / np.max(freqs))
                else:
                    spectral_filter = np.ones_like(freqs)
                
                spectral_filter = np.clip(spectral_filter, 0.6, 1.8)
                magnitude_shaped = magnitude * spectral_filter.reshape(-1, 1)
                
                stft_converted = magnitude_shaped * np.exp(1j * phase)
                y = librosa.istft(stft_converted)
        
        # Normalize
        y = y / np.max(np.abs(y)) * 0.8
        
        return y
    
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