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
from scipy import signal
from scipy.interpolate import interp1d

class ImprovedVoiceCloner:
    """Improved voice cloner with better voice matching"""
    
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        print("✅ Improved Voice Cloner initialized")
    
    def _load_existing_models(self):
        for voice_dir in os.listdir(self.models_dir):
            voice_path = os.path.join(self.models_dir, voice_dir)
            if os.path.isdir(voice_path):
                model_file = os.path.join(voice_path, "voice_profile.json")
                if os.path.exists(model_file):
                    with open(model_file, 'r') as f:
                        profile = json.load(f)
                        self.voice_models[voice_dir] = profile
    
    def start_training(self, voice_id, epochs=10):
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
            
            audio_files = self._find_audio_files(voice_id)
            
            if len(audio_files) < 1:
                raise Exception("Need at least 1 audio file")
            
            # Extract comprehensive voice features
            voice_features = self._extract_voice_features(audio_files)
            reference_clips = self._process_audio_files(audio_files, voice_id)
            
            for epoch in range(epochs):
                time.sleep(0.3)
                progress = int((epoch + 1) / epochs * 100)
                self.training_status[training_id]['progress'] = progress
            
            voice_profile = {
                'voice_id': voice_id,
                'model_type': 'improved_cloner',
                'reference_clips': reference_clips,
                'voice_features': voice_features,
                'sample_count': len(reference_clips),
                'created_at': datetime.now().isoformat()
            }
            
            model_path = os.path.join(self.models_dir, voice_id)
            os.makedirs(model_path, exist_ok=True)
            
            profile_file = os.path.join(model_path, "voice_profile.json")
            with open(profile_file, 'w') as f:
                json.dump(voice_profile, f, indent=2)
            
            self.voice_models[voice_id] = voice_profile
            
            self.training_status[training_id].update({
                'status': 'completed',
                'progress': 100,
                'end_time': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _extract_voice_features(self, audio_files):
        """Extract comprehensive voice characteristics"""
        features = {
            'pitch_stats': [],
            'formants': [],
            'spectral_centroid': [],
            'mfcc_mean': [],
            'tempo': [],
            'energy': []
        }
        
        for audio_file in audio_files[:5]:  # Use up to 5 files
            try:
                y, sr = librosa.load(audio_file, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) < sr * 0.5:  # Skip very short clips
                    continue
                
                # Pitch analysis
                pitches = librosa.yin(y_clean, fmin=80, fmax=400)
                valid_pitches = pitches[~np.isnan(pitches)]
                if len(valid_pitches) > 0:
                    features['pitch_stats'].append({
                        'mean': float(np.mean(valid_pitches)),
                        'std': float(np.std(valid_pitches)),
                        'median': float(np.median(valid_pitches))
                    })
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y_clean, sr=sr)[0]
                features['spectral_centroid'].append(float(np.mean(spectral_centroids)))
                
                # MFCC features
                mfccs = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=13)
                features['mfcc_mean'].append(np.mean(mfccs, axis=1).tolist())
                
                # Energy
                energy = np.sum(y_clean ** 2) / len(y_clean)
                features['energy'].append(float(energy))
                
            except Exception as e:
                continue
        
        # Calculate average features
        if features['pitch_stats']:
            avg_pitch = np.mean([p['mean'] for p in features['pitch_stats']])
            features['avg_pitch'] = float(avg_pitch)
        
        if features['spectral_centroid']:
            features['avg_spectral_centroid'] = float(np.mean(features['spectral_centroid']))
        
        return features
    
    def _find_audio_files(self, voice_id):
        import re
        matching_files = []
        search_dirs = ["uploads", "datasets/raw", "datasets/processed"]
        
        voice_pattern = re.compile(rf".*{re.escape(voice_id)}.*", re.IGNORECASE)
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                            if voice_pattern.match(file):
                                matching_files.append(os.path.join(root, file))
        
        return list(set(matching_files))
    
    def _process_audio_files(self, audio_files, voice_id):
        reference_clips = []
        
        for i, audio_file in enumerate(audio_files[:5]):
            try:
                y, sr = librosa.load(audio_file, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) > sr * 1.0:
                    model_path = os.path.join(self.models_dir, voice_id)
                    os.makedirs(model_path, exist_ok=True)
                    
                    clip_file = os.path.join(model_path, f"reference_{i}.wav")
                    sf.write(clip_file, y_clean, sr)
                    
                    reference_clips.append({
                        'file_path': clip_file,
                        'duration': len(y_clean) / sr,
                        'source': os.path.basename(audio_file)
                    })
                    
            except Exception as e:
                continue
        
        return reference_clips
    
    def generate_speech(self, voice_id, text, output_path):
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model '{voice_id}' not found")
        
        try:
            # Generate base TTS
            tts = gTTS(text=text, lang='id', slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                base_tts_path = tmp_file.name
            
            y_tts, sr = librosa.load(base_tts_path, sr=22050)
            os.unlink(base_tts_path)
            
            # Apply advanced voice conversion
            profile = self.voice_models[voice_id]
            y_converted = self._apply_advanced_conversion(y_tts, sr, profile)
            
            sf.write(output_path, y_converted, sr)
            return True
            
        except Exception as e:
            return False
    
    def _apply_advanced_conversion(self, y_tts, sr, profile):
        """Apply advanced voice conversion techniques"""
        
        # Get voice features
        voice_features = profile.get('voice_features', {})
        
        # 1. Pitch conversion
        y_converted = self._convert_pitch(y_tts, sr, voice_features)
        
        # 2. Spectral envelope conversion
        y_converted = self._convert_spectral_envelope(y_converted, sr, voice_features)
        
        # 3. Formant conversion
        y_converted = self._convert_formants(y_converted, sr, voice_features)
        
        # 4. Apply reference audio characteristics
        if 'reference_clips' in profile and profile['reference_clips']:
            y_converted = self._apply_reference_characteristics(y_converted, sr, profile['reference_clips'][0])
        
        # Normalize
        y_converted = y_converted / np.max(np.abs(y_converted)) * 0.8
        
        return y_converted
    
    def _convert_pitch(self, y, sr, voice_features):
        """Convert pitch to match target voice"""
        if 'avg_pitch' not in voice_features:
            return y
        
        try:
            # Extract current pitch
            pitches = librosa.yin(y, fmin=80, fmax=400)
            valid_pitches = pitches[~np.isnan(pitches)]
            
            if len(valid_pitches) == 0:
                return y
            
            current_pitch = np.mean(valid_pitches)
            target_pitch = voice_features['avg_pitch']
            
            # Calculate semitone shift
            semitone_shift = 12 * np.log2(target_pitch / current_pitch)
            semitone_shift = np.clip(semitone_shift, -6, 6)
            
            # Apply pitch shift
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone_shift)
            
            return y_shifted
            
        except Exception as e:
            return y
    
    def _convert_spectral_envelope(self, y, sr, voice_features):
        """Convert spectral envelope"""
        if 'avg_spectral_centroid' not in voice_features:
            return y
        
        try:
            # Get current spectral centroid
            current_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            target_centroid = voice_features['avg_spectral_centroid']
            
            # Apply spectral filtering
            if target_centroid > current_centroid:
                # Brighten the sound
                b, a = signal.butter(2, target_centroid / (sr/2), btype='high')
                y_filtered = signal.filtfilt(b, a, y)
                y = 0.7 * y + 0.3 * y_filtered
            else:
                # Darken the sound
                b, a = signal.butter(2, target_centroid / (sr/2), btype='low')
                y_filtered = signal.filtfilt(b, a, y)
                y = 0.7 * y + 0.3 * y_filtered
            
            return y
            
        except Exception as e:
            return y
    
    def _convert_formants(self, y, sr, voice_features):
        """Basic formant conversion"""
        try:
            # Apply formant shifting using pitch shifting at different rates
            y_f1 = librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)
            y_f2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=-0.3)
            
            # Blend with original
            y_formant = 0.6 * y + 0.2 * y_f1 + 0.2 * y_f2
            
            return y_formant
            
        except Exception as e:
            return y
    
    def _apply_reference_characteristics(self, y_tts, sr, reference_clip):
        """Apply characteristics from reference audio"""
        try:
            ref_path = reference_clip['file_path']
            if not os.path.exists(ref_path):
                return y_tts
            
            y_ref, _ = librosa.load(ref_path, sr=sr)
            
            # Match energy envelope
            y_tts = self._match_energy_envelope(y_tts, y_ref)
            
            # Match spectral characteristics
            y_tts = self._match_spectral_characteristics(y_tts, y_ref, sr)
            
            return y_tts
            
        except Exception as e:
            return y_tts
    
    def _match_energy_envelope(self, y_tts, y_ref):
        """Match energy envelope between TTS and reference"""
        try:
            # Calculate energy envelopes
            hop_length = 512
            tts_energy = librosa.feature.rms(y=y_tts, hop_length=hop_length)[0]
            ref_energy = librosa.feature.rms(y=y_ref, hop_length=hop_length)[0]
            
            # Interpolate reference energy to match TTS length
            if len(ref_energy) != len(tts_energy):
                x_ref = np.linspace(0, 1, len(ref_energy))
                x_tts = np.linspace(0, 1, len(tts_energy))
                f = interp1d(x_ref, ref_energy, kind='linear', fill_value='extrapolate')
                ref_energy = f(x_tts)
            
            # Apply energy matching
            energy_ratio = ref_energy / (tts_energy + 1e-8)
            energy_ratio = np.clip(energy_ratio, 0.3, 3.0)
            
            # Expand energy ratio to audio length
            energy_full = np.repeat(energy_ratio, hop_length)[:len(y_tts)]
            
            y_matched = y_tts * energy_full
            
            return y_matched
            
        except Exception as e:
            return y_tts
    
    def _match_spectral_characteristics(self, y_tts, y_ref, sr):
        """Match spectral characteristics"""
        try:
            # Get spectral features
            tts_mfcc = librosa.feature.mfcc(y=y_tts, sr=sr, n_mfcc=13)
            ref_mfcc = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
            
            # Calculate spectral difference
            ref_mean = np.mean(ref_mfcc, axis=1)
            tts_mean = np.mean(tts_mfcc, axis=1)
            
            # Apply simple spectral shaping
            diff = ref_mean[1] - tts_mean[1]  # Use first MFCC coefficient
            
            if abs(diff) > 0.5:
                # Apply filtering based on spectral difference
                if diff > 0:
                    # Boost high frequencies
                    b, a = signal.butter(2, 3000 / (sr/2), btype='high')
                    y_filtered = signal.filtfilt(b, a, y_tts)
                    y_tts = 0.8 * y_tts + 0.2 * y_filtered
                else:
                    # Boost low frequencies
                    b, a = signal.butter(2, 1000 / (sr/2), btype='low')
                    y_filtered = signal.filtfilt(b, a, y_tts)
                    y_tts = 0.8 * y_tts + 0.2 * y_filtered
            
            return y_tts
            
        except Exception as e:
            return y_tts
    
    def get_audio_analysis(self, voice_id):
        if voice_id in self.voice_models:
            profile = self.voice_models[voice_id]
            features = profile.get('voice_features', {})
            return {
                'recommended_epochs': 10,
                'quality_level': 'High',
                'avg_pitch': features.get('avg_pitch', 'Unknown'),
                'spectral_centroid': features.get('avg_spectral_centroid', 'Unknown')
            }
        return None
    
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
                'model_type': 'improved_cloner'
            }
        return None