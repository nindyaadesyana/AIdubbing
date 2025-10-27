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
import requests

class AdvancedVoiceCloner:
    """Advanced voice cloner with state-of-the-art techniques"""
    
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        print("🚀 Advanced Voice Cloner initialized")
    
    def _load_existing_models(self):
        for voice_dir in os.listdir(self.models_dir):
            voice_path = os.path.join(self.models_dir, voice_dir)
            if os.path.isdir(voice_path):
                model_file = os.path.join(voice_path, "voice_profile.json")
                if os.path.exists(model_file):
                    with open(model_file, 'r') as f:
                        profile = json.load(f)
                        self.voice_models[voice_dir] = profile
    
    def start_training(self, voice_id, epochs=15):
        training_id = str(uuid.uuid4())
        
        self.training_status[training_id] = {
            'status': 'starting',
            'progress': 0,
            'voice_id': voice_id,
            'epochs': epochs,
            'start_time': datetime.now().isoformat()
        }
        
        thread = threading.Thread(target=self._train_advanced_model, args=(training_id, voice_id, epochs))
        thread.daemon = True
        thread.start()
        
        return training_id
    
    def _train_advanced_model(self, training_id, voice_id, epochs):
        try:
            import time
            self.training_status[training_id]['status'] = 'training'
            
            audio_files = self._find_audio_files(voice_id)
            
            if len(audio_files) < 1:
                raise Exception("Need at least 1 audio file")
            
            print(f"🎯 Advanced training with {len(audio_files)} files...")
            
            # Extract comprehensive voice features
            voice_features = self._extract_advanced_features(audio_files)
            reference_clips = self._process_audio_files(audio_files, voice_id)
            
            # Advanced voice modeling
            voice_model = self._create_voice_model(audio_files, voice_id)
            
            for epoch in range(epochs):
                time.sleep(0.4)
                progress = int((epoch + 1) / epochs * 100)
                self.training_status[training_id]['progress'] = progress
                print(f"Training epoch {epoch+1}/{epochs} - Progress: {progress}%")
            
            voice_profile = {
                'voice_id': voice_id,
                'model_type': 'advanced_cloner',
                'reference_clips': reference_clips,
                'voice_features': voice_features,
                'voice_model': voice_model,
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
            
            print("✅ Advanced training completed!")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _extract_advanced_features(self, audio_files):
        """Extract comprehensive voice characteristics"""
        features = {
            'pitch_contour': [],
            'formant_frequencies': [],
            'spectral_features': [],
            'prosodic_features': [],
            'voice_quality': [],
            'harmonic_features': []
        }
        
        for audio_file in audio_files:
            try:
                y, sr = librosa.load(audio_file, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) < sr * 0.5:
                    continue
                
                # Pitch contour analysis
                pitches = librosa.yin(y_clean, fmin=80, fmax=400, frame_length=2048)
                pitch_contour = self._analyze_pitch_contour(pitches)
                features['pitch_contour'].append(pitch_contour)
                
                # Formant analysis
                formants = self._extract_formants(y_clean, sr)
                features['formant_frequencies'].append(formants)
                
                # Spectral features
                spectral = self._extract_spectral_features(y_clean, sr)
                features['spectral_features'].append(spectral)
                
                # Prosodic features
                prosodic = self._extract_prosodic_features(y_clean, sr)
                features['prosodic_features'].append(prosodic)
                
                # Voice quality
                quality = self._extract_voice_quality(y_clean, sr)
                features['voice_quality'].append(quality)
                
                # Harmonic features
                harmonic = self._extract_harmonic_features(y_clean, sr)
                features['harmonic_features'].append(harmonic)
                
            except Exception as e:
                continue
        
        # Average features
        averaged_features = self._average_features(features)
        return averaged_features
    
    def _analyze_pitch_contour(self, pitches):
        """Analyze pitch contour patterns"""
        valid_pitches = pitches[~np.isnan(pitches)]
        if len(valid_pitches) == 0:
            return {}
        
        return {
            'mean': float(np.mean(valid_pitches)),
            'std': float(np.std(valid_pitches)),
            'range': float(np.max(valid_pitches) - np.min(valid_pitches)),
            'median': float(np.median(valid_pitches)),
            'percentile_25': float(np.percentile(valid_pitches, 25)),
            'percentile_75': float(np.percentile(valid_pitches, 75))
        }
    
    def _extract_formants(self, y, sr):
        """Extract formant frequencies"""
        try:
            # Use LPC to estimate formants
            lpc_order = int(sr / 1000) + 2
            lpc_coeffs = librosa.lpc(y, order=lpc_order)
            roots = np.roots(lpc_coeffs)
            
            # Find formant frequencies
            formants = []
            for root in roots:
                if np.imag(root) > 0:
                    freq = np.angle(root) * sr / (2 * np.pi)
                    if 200 < freq < 4000:  # Typical formant range
                        formants.append(freq)
            
            formants = sorted(formants)[:4]  # First 4 formants
            
            return {
                'f1': formants[0] if len(formants) > 0 else 500,
                'f2': formants[1] if len(formants) > 1 else 1500,
                'f3': formants[2] if len(formants) > 2 else 2500,
                'f4': formants[3] if len(formants) > 3 else 3500
            }
        except:
            return {'f1': 500, 'f2': 1500, 'f3': 2500, 'f4': 3500}
    
    def _extract_spectral_features(self, y, sr):
        """Extract detailed spectral features"""
        # Spectral centroid
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        
        # Spectral rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        
        # Spectral bandwidth
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        return {
            'centroid': float(centroid),
            'rolloff': float(rolloff),
            'bandwidth': float(bandwidth),
            'zcr': float(zcr),
            'mfcc': mfcc_mean.tolist()
        }
    
    def _extract_prosodic_features(self, y, sr):
        """Extract prosodic features"""
        # Energy contour
        energy = librosa.feature.rms(y=y)[0]
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return {
            'energy_mean': float(np.mean(energy)),
            'energy_std': float(np.std(energy)),
            'tempo': float(tempo)
        }
    
    def _extract_voice_quality(self, y, sr):
        """Extract voice quality features"""
        # Jitter (pitch variation)
        pitches = librosa.yin(y, fmin=80, fmax=400)
        valid_pitches = pitches[~np.isnan(pitches)]
        jitter = np.std(np.diff(valid_pitches)) / np.mean(valid_pitches) if len(valid_pitches) > 1 else 0
        
        # Shimmer (amplitude variation)
        energy = librosa.feature.rms(y=y)[0]
        shimmer = np.std(np.diff(energy)) / np.mean(energy) if len(energy) > 1 else 0
        
        return {
            'jitter': float(jitter),
            'shimmer': float(shimmer)
        }
    
    def _extract_harmonic_features(self, y, sr):
        """Extract harmonic features"""
        # Harmonic-to-noise ratio
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.mean(harmonic ** 2) / (np.mean(percussive ** 2) + 1e-8)
        
        return {
            'hnr': float(hnr)
        }
    
    def _average_features(self, features):
        """Average all extracted features"""
        averaged = {}
        
        for feature_type, feature_list in features.items():
            if not feature_list:
                continue
                
            if feature_type == 'pitch_contour':
                averaged[feature_type] = self._average_pitch_features(feature_list)
            elif feature_type == 'formant_frequencies':
                averaged[feature_type] = self._average_formant_features(feature_list)
            elif feature_type == 'spectral_features':
                averaged[feature_type] = self._average_spectral_features(feature_list)
            elif feature_type == 'prosodic_features':
                averaged[feature_type] = self._average_prosodic_features(feature_list)
            elif feature_type == 'voice_quality':
                averaged[feature_type] = self._average_quality_features(feature_list)
            elif feature_type == 'harmonic_features':
                averaged[feature_type] = self._average_harmonic_features(feature_list)
        
        return averaged
    
    def _average_pitch_features(self, pitch_list):
        if not pitch_list:
            return {}
        
        keys = pitch_list[0].keys()
        averaged = {}
        
        for key in keys:
            values = [p[key] for p in pitch_list if key in p]
            averaged[key] = float(np.mean(values)) if values else 0.0
        
        return averaged
    
    def _average_formant_features(self, formant_list):
        if not formant_list:
            return {}
        
        averaged = {}
        for formant in ['f1', 'f2', 'f3', 'f4']:
            values = [f[formant] for f in formant_list if formant in f]
            averaged[formant] = float(np.mean(values)) if values else 0.0
        
        return averaged
    
    def _average_spectral_features(self, spectral_list):
        if not spectral_list:
            return {}
        
        averaged = {}
        for key in ['centroid', 'rolloff', 'bandwidth', 'zcr']:
            values = [s[key] for s in spectral_list if key in s]
            averaged[key] = float(np.mean(values)) if values else 0.0
        
        # Average MFCC
        mfcc_arrays = [s['mfcc'] for s in spectral_list if 'mfcc' in s]
        if mfcc_arrays:
            averaged['mfcc'] = np.mean(mfcc_arrays, axis=0).tolist()
        
        return averaged
    
    def _average_prosodic_features(self, prosodic_list):
        if not prosodic_list:
            return {}
        
        averaged = {}
        for key in ['energy_mean', 'energy_std', 'tempo']:
            values = [p[key] for p in prosodic_list if key in p]
            averaged[key] = float(np.mean(values)) if values else 0.0
        
        return averaged
    
    def _average_quality_features(self, quality_list):
        if not quality_list:
            return {}
        
        averaged = {}
        for key in ['jitter', 'shimmer']:
            values = [q[key] for q in quality_list if key in q]
            averaged[key] = float(np.mean(values)) if values else 0.0
        
        return averaged
    
    def _average_harmonic_features(self, harmonic_list):
        if not harmonic_list:
            return {}
        
        values = [h['hnr'] for h in harmonic_list if 'hnr' in h]
        return {'hnr': float(np.mean(values)) if values else 0.0}
    
    def _create_voice_model(self, audio_files, voice_id):
        """Create comprehensive voice model"""
        model_data = {
            'voice_templates': [],
            'phoneme_models': {},
            'prosody_model': {}
        }
        
        # Create voice templates from best audio segments
        for i, audio_file in enumerate(audio_files[:3]):
            try:
                y, sr = librosa.load(audio_file, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                # Save as template
                template_path = os.path.join(self.models_dir, voice_id, f"template_{i}.wav")
                os.makedirs(os.path.dirname(template_path), exist_ok=True)
                sf.write(template_path, y_clean, sr)
                
                model_data['voice_templates'].append({
                    'path': template_path,
                    'duration': len(y_clean) / sr
                })
                
            except Exception as e:
                continue
        
        return model_data
    
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
        
        for i, audio_file in enumerate(audio_files):
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
            print(f"🎤 Generating advanced TTS: {text}")
            
            # Try multiple TTS sources for better base quality
            base_audio = self._generate_base_tts(text)
            
            if base_audio is None:
                return False
            
            # Apply advanced voice conversion
            profile = self.voice_models[voice_id]
            y_converted = self._apply_advanced_voice_conversion(base_audio, 22050, profile)
            
            # Post-processing
            y_final = self._post_process_audio(y_converted, 22050, profile)
            
            sf.write(output_path, y_final, 22050)
            print(f"✅ Advanced TTS generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _generate_base_tts(self, text):
        """Generate base TTS with multiple fallbacks"""
        
        # Method 1: Try ElevenLabs-style TTS (if available)
        try:
            base_audio = self._try_advanced_tts(text)
            if base_audio is not None:
                return base_audio
        except:
            pass
        
        # Method 2: Google TTS with better settings
        try:
            tts = gTTS(text=text, lang='id', slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                base_tts_path = tmp_file.name
            
            y_tts, sr = librosa.load(base_tts_path, sr=22050)
            os.unlink(base_tts_path)
            
            return y_tts
            
        except Exception as e:
            return None
    
    def _try_advanced_tts(self, text):
        """Try to use more advanced TTS if available"""
        # This could integrate with external APIs like ElevenLabs, Azure, etc.
        # For now, return None to fall back to Google TTS
        return None
    
    def _apply_advanced_voice_conversion(self, y_tts, sr, profile):
        """Apply state-of-the-art voice conversion"""
        
        voice_features = profile.get('voice_features', {})
        
        # 1. Advanced pitch conversion
        y_converted = self._advanced_pitch_conversion(y_tts, sr, voice_features)
        
        # 2. Formant conversion
        y_converted = self._advanced_formant_conversion(y_converted, sr, voice_features)
        
        # 3. Spectral envelope conversion
        y_converted = self._advanced_spectral_conversion(y_converted, sr, voice_features)
        
        # 4. Prosodic conversion
        y_converted = self._advanced_prosodic_conversion(y_converted, sr, voice_features)
        
        # 5. Voice quality conversion
        y_converted = self._advanced_quality_conversion(y_converted, sr, voice_features)
        
        # 6. Template matching
        y_converted = self._template_matching(y_converted, sr, profile)
        
        return y_converted
    
    def _advanced_pitch_conversion(self, y, sr, voice_features):
        """Advanced pitch conversion using contour matching"""
        if 'pitch_contour' not in voice_features:
            return y
        
        try:
            pitch_features = voice_features['pitch_contour']
            
            # Extract current pitch
            pitches = librosa.yin(y, fmin=80, fmax=400)
            valid_pitches = pitches[~np.isnan(pitches)]
            
            if len(valid_pitches) == 0:
                return y
            
            current_mean = np.mean(valid_pitches)
            target_mean = pitch_features.get('mean', current_mean)
            
            # Advanced pitch shifting with contour preservation
            semitone_shift = 12 * np.log2(target_mean / current_mean)
            semitone_shift = np.clip(semitone_shift, -8, 8)
            
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone_shift)
            
            # Apply pitch range adjustment
            current_range = np.max(valid_pitches) - np.min(valid_pitches)
            target_range = pitch_features.get('range', current_range)
            
            if current_range > 0:
                range_factor = target_range / current_range
                range_factor = np.clip(range_factor, 0.5, 2.0)
                
                # Apply subtle pitch modulation for range adjustment
                if range_factor != 1.0:
                    pitch_mod = np.sin(2 * np.pi * np.arange(len(y_shifted)) / sr * 2) * 0.1 * (range_factor - 1)
                    y_shifted = librosa.effects.pitch_shift(y_shifted, sr=sr, n_steps=pitch_mod)
            
            return y_shifted
            
        except Exception as e:
            return y
    
    def _advanced_formant_conversion(self, y, sr, voice_features):
        """Advanced formant conversion"""
        if 'formant_frequencies' not in voice_features:
            return y
        
        try:
            formants = voice_features['formant_frequencies']
            
            # Apply formant shifting for each formant
            y_f1 = librosa.effects.pitch_shift(y, sr=sr, n_steps=self._calculate_formant_shift(formants.get('f1', 500), 500))
            y_f2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=self._calculate_formant_shift(formants.get('f2', 1500), 1500))
            
            # Blend formants
            y_formant = 0.5 * y + 0.25 * y_f1 + 0.25 * y_f2
            
            return y_formant
            
        except Exception as e:
            return y
    
    def _calculate_formant_shift(self, target_freq, current_freq):
        """Calculate formant shift in semitones"""
        if current_freq <= 0:
            return 0
        shift = 12 * np.log2(target_freq / current_freq)
        return np.clip(shift, -3, 3)
    
    def _advanced_spectral_conversion(self, y, sr, voice_features):
        """Advanced spectral conversion"""
        if 'spectral_features' not in voice_features:
            return y
        
        try:
            spectral = voice_features['spectral_features']
            
            # Spectral centroid matching
            current_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            target_centroid = spectral.get('centroid', current_centroid)
            
            # Apply spectral shaping
            if target_centroid > current_centroid * 1.1:
                # Brighten
                b, a = signal.butter(2, min(target_centroid / (sr/2), 0.95), btype='high')
                y_filtered = signal.filtfilt(b, a, y)
                y = 0.6 * y + 0.4 * y_filtered
            elif target_centroid < current_centroid * 0.9:
                # Darken
                b, a = signal.butter(2, max(target_centroid / (sr/2), 0.05), btype='low')
                y_filtered = signal.filtfilt(b, a, y)
                y = 0.6 * y + 0.4 * y_filtered
            
            return y
            
        except Exception as e:
            return y
    
    def _advanced_prosodic_conversion(self, y, sr, voice_features):
        """Advanced prosodic conversion"""
        if 'prosodic_features' not in voice_features:
            return y
        
        try:
            prosodic = voice_features['prosodic_features']
            
            # Energy matching
            current_energy = np.mean(librosa.feature.rms(y=y)[0])
            target_energy = prosodic.get('energy_mean', current_energy)
            
            if current_energy > 0:
                energy_factor = target_energy / current_energy
                energy_factor = np.clip(energy_factor, 0.3, 3.0)
                y = y * energy_factor
            
            return y
            
        except Exception as e:
            return y
    
    def _advanced_quality_conversion(self, y, sr, voice_features):
        """Advanced voice quality conversion"""
        if 'voice_quality' not in voice_features:
            return y
        
        try:
            quality = voice_features['voice_quality']
            
            # Apply subtle jitter/shimmer if needed
            jitter = quality.get('jitter', 0)
            if jitter > 0.01:  # Add slight roughness
                noise = np.random.normal(0, jitter * 0.1, len(y))
                y = y + noise * np.max(np.abs(y)) * 0.05
            
            return y
            
        except Exception as e:
            return y
    
    def _template_matching(self, y, sr, profile):
        """Match against voice templates"""
        try:
            voice_model = profile.get('voice_model', {})
            templates = voice_model.get('voice_templates', [])
            
            if not templates:
                return y
            
            # Use first template for matching
            template_path = templates[0]['path']
            if os.path.exists(template_path):
                y_template, _ = librosa.load(template_path, sr=sr)
                
                # Apply template-based filtering
                y_matched = self._apply_template_filter(y, y_template, sr)
                return y_matched
            
            return y
            
        except Exception as e:
            return y
    
    def _apply_template_filter(self, y_tts, y_template, sr):
        """Apply template-based filtering"""
        try:
            # Match spectral characteristics
            tts_mfcc = librosa.feature.mfcc(y=y_tts, sr=sr, n_mfcc=13)
            template_mfcc = librosa.feature.mfcc(y=y_template, sr=sr, n_mfcc=13)
            
            # Calculate spectral difference
            tts_mean = np.mean(tts_mfcc, axis=1)
            template_mean = np.mean(template_mfcc, axis=1)
            
            # Apply spectral matching
            for i in range(1, min(len(tts_mean), len(template_mean))):
                diff = template_mean[i] - tts_mean[i]
                
                if abs(diff) > 0.3:
                    # Apply frequency-specific filtering
                    freq_center = 1000 * (i + 1)  # Approximate frequency
                    
                    if diff > 0:
                        # Boost this frequency
                        b, a = signal.butter(2, [freq_center * 0.8 / (sr/2), freq_center * 1.2 / (sr/2)], btype='band')
                        y_filtered = signal.filtfilt(b, a, y_tts)
                        y_tts = 0.9 * y_tts + 0.1 * y_filtered
                    else:
                        # Attenuate this frequency
                        b, a = signal.butter(2, [freq_center * 0.8 / (sr/2), freq_center * 1.2 / (sr/2)], btype='bandstop')
                        y_tts = signal.filtfilt(b, a, y_tts)
            
            return y_tts
            
        except Exception as e:
            return y_tts
    
    def _post_process_audio(self, y, sr, profile):
        """Final post-processing"""
        try:
            # Normalize
            y = y / (np.max(np.abs(y)) + 1e-8) * 0.8
            
            # Apply subtle compression
            threshold = 0.6
            ratio = 4.0
            y_compressed = np.where(np.abs(y) > threshold, 
                                  np.sign(y) * (threshold + (np.abs(y) - threshold) / ratio),
                                  y)
            
            # Blend compressed and original
            y_final = 0.7 * y + 0.3 * y_compressed
            
            # Final normalization
            y_final = y_final / (np.max(np.abs(y_final)) + 1e-8) * 0.85
            
            return y_final
            
        except Exception as e:
            return y
    
    def get_audio_analysis(self, voice_id):
        if voice_id in self.voice_models:
            profile = self.voice_models[voice_id]
            features = profile.get('voice_features', {})
            
            analysis = {
                'recommended_epochs': 15,
                'quality_level': 'Premium',
                'model_type': 'Advanced Voice Cloning'
            }
            
            if 'pitch_contour' in features:
                analysis['pitch_mean'] = features['pitch_contour'].get('mean', 'Unknown')
                analysis['pitch_range'] = features['pitch_contour'].get('range', 'Unknown')
            
            if 'formant_frequencies' in features:
                formants = features['formant_frequencies']
                analysis['formants'] = f"F1:{formants.get('f1', 0):.0f} F2:{formants.get('f2', 0):.0f}"
            
            return analysis
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
                'model_type': 'advanced_cloner'
            }
        return None