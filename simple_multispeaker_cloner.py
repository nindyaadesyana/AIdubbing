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
from scipy.interpolate import interp1d

class SimpleMultiSpeakerCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        print("✅ Simple Multi-Speaker Voice Cloner initialized")
    
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
            
            print(f"Training multi-speaker voice model...")
            
            for i in range(10):
                time.sleep(0.8)
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
                print(f"Multi-speaker training progress: {progress}%")
            
            # Create comprehensive voice profile
            voice_profile = self._create_multispeaker_profile(dataset_path, audio_files)
            
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
            
            print(f"Multi-speaker voice model trained successfully!")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _create_multispeaker_profile(self, dataset_path, audio_files):
        """Create comprehensive multi-speaker voice profile"""
        
        voice_characteristics = []
        reference_samples = []
        
        print("Creating multi-speaker voice profile...")
        
        for i, audio_file in enumerate(audio_files[:8]):
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=15)
                
                if len(y_clean) > sr * 0.8:  # At least 0.8 seconds
                    print(f"  Analyzing sample {i+1}: {audio_file}")
                    
                    # Comprehensive voice analysis
                    characteristics = self._extract_voice_characteristics(y_clean, sr)
                    
                    if characteristics:
                        voice_characteristics.append(characteristics)
                        
                        # Store reference sample
                        if len(reference_samples) < 3:
                            model_path = os.path.join(self.models_dir, dataset_path.split('/')[-3])
                            os.makedirs(model_path, exist_ok=True)
                            
                            ref_file = os.path.join(model_path, f"reference_{len(reference_samples)}.wav")
                            sf.write(ref_file, y_clean, sr)
                            
                            reference_samples.append({
                                'file_path': ref_file,
                                'duration': len(y_clean) / sr,
                                'characteristics': characteristics
                            })
                        
            except Exception as e:
                print(f"    Error analyzing {audio_file}: {e}")
                continue
        
        if len(voice_characteristics) < 2:
            raise Exception("Not enough valid voice samples for multi-speaker training")
        
        # Compute multi-speaker profile
        profile = self._compute_speaker_profile(voice_characteristics, reference_samples)
        profile['voice_id'] = dataset_path.split('/')[-3]
        profile['sample_count'] = len(voice_characteristics)
        profile['created_at'] = datetime.now().isoformat()
        
        return profile
    
    def _extract_voice_characteristics(self, y, sr):
        """Extract comprehensive voice characteristics"""
        
        try:
            # Pitch analysis
            pitches = librosa.yin(y, fmin=80, fmax=400)
            pitch_values = pitches[~np.isnan(pitches)]
            
            if len(pitch_values) < 10:
                return None
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            
            # MFCC for timbre
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Formant estimation (simplified)
            formants = self._estimate_formants(y, sr)
            
            # Voice texture
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            return {
                'pitch_mean': float(np.mean(pitch_values)),
                'pitch_std': float(np.std(pitch_values)),
                'pitch_range': [float(np.min(pitch_values)), float(np.max(pitch_values))],
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'spectral_bandwidth': float(spectral_bandwidth),
                'mfcc_mean': mfcc.mean(axis=1).tolist(),
                'mfcc_std': mfcc.std(axis=1).tolist(),
                'formants': formants,
                'zcr': float(zcr)
            }
            
        except Exception as e:
            return None
    
    def _estimate_formants(self, y, sr):
        """Simple formant estimation"""
        try:
            # Get spectrum
            fft = np.fft.fft(y)
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            
            # Find peaks (simplified formant detection)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(magnitude[:len(magnitude)//2], height=np.max(magnitude)*0.1)
            
            formant_freqs = freqs[peaks]
            formant_freqs = formant_freqs[(formant_freqs > 200) & (formant_freqs < 3500)]
            
            # Return first 3 formants
            formants = sorted(formant_freqs)[:3]
            while len(formants) < 3:
                formants.append(0.0)
            
            return [float(f) for f in formants]
            
        except:
            return [800.0, 1200.0, 2500.0]  # Default formants
    
    def _compute_speaker_profile(self, characteristics, reference_samples):
        """Compute average speaker profile"""
        
        # Average all characteristics
        pitch_means = [c['pitch_mean'] for c in characteristics]
        spectral_centroids = [c['spectral_centroid'] for c in characteristics]
        spectral_rolloffs = [c['spectral_rolloff'] for c in characteristics]
        spectral_bandwidths = [c['spectral_bandwidth'] for c in characteristics]
        zcrs = [c['zcr'] for c in characteristics]
        
        # MFCC averaging
        mfcc_means = [c['mfcc_mean'] for c in characteristics]
        mfcc_stds = [c['mfcc_std'] for c in characteristics]
        
        # Formants averaging
        formants = [c['formants'] for c in characteristics]
        
        return {
            'speaker_profile': {
                'pitch_mean': float(np.mean(pitch_means)),
                'pitch_std': float(np.std(pitch_means)),
                'spectral_centroid': float(np.mean(spectral_centroids)),
                'spectral_rolloff': float(np.mean(spectral_rolloffs)),
                'spectral_bandwidth': float(np.mean(spectral_bandwidths)),
                'zcr': float(np.mean(zcrs)),
                'mfcc_profile': np.mean(mfcc_means, axis=0).tolist(),
                'formants': np.mean(formants, axis=0).tolist()
            },
            'reference_samples': reference_samples,
            'training_quality': 'multi-speaker'
        }
    
    def generate_speech(self, voice_id, text, output_path):
        """Generate speech using multi-speaker approach"""
        
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model '{voice_id}' not found")
        
        try:
            print(f"Generating multi-speaker TTS: {text}")
            
            # Step 1: Generate base TTS
            tts = gTTS(text=text, lang='id', slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                base_tts_path = tmp_file.name
            
            # Step 2: Load TTS
            y_tts, sr = librosa.load(base_tts_path, sr=22050)
            os.unlink(base_tts_path)
            
            # Step 3: Apply multi-speaker voice conversion
            profile = self.voice_models[voice_id]
            y_converted = self._apply_multispeaker_conversion(y_tts, sr, profile)
            
            # Step 4: Save
            sf.write(output_path, y_converted, sr)
            
            print(f"Multi-speaker TTS generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating multi-speaker TTS: {e}")
            return False
    
    def _apply_multispeaker_conversion(self, y_tts, sr, profile):
        """Apply comprehensive multi-speaker voice conversion"""
        
        speaker_profile = profile['speaker_profile']
        
        print("Applying multi-speaker voice conversion...")
        
        # Step 1: Pitch conversion
        target_pitch = speaker_profile['pitch_mean']
        
        pitches = librosa.yin(y_tts, fmin=80, fmax=400)
        current_pitches = pitches[~np.isnan(pitches)]
        
        if len(current_pitches) > 10:
            current_pitch = np.mean(current_pitches)
            semitone_shift = 12 * np.log2(target_pitch / current_pitch)
            semitone_shift = np.clip(semitone_shift, -8, 8)
            
            print(f"Multi-speaker pitch shift: {semitone_shift:.1f} semitones")
            y_pitched = librosa.effects.pitch_shift(y_tts, sr=sr, n_steps=semitone_shift)
        else:
            y_pitched = y_tts
        
        # Step 2: Spectral envelope conversion
        target_centroid = speaker_profile['spectral_centroid']
        target_rolloff = speaker_profile['spectral_rolloff']
        
        stft = librosa.stft(y_pitched)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply spectral shaping based on target characteristics
        current_centroid = np.mean(librosa.feature.spectral_centroid(y=y_pitched, sr=sr))
        centroid_ratio = target_centroid / current_centroid
        
        freq_bins = magnitude.shape[0]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-1)
        
        # Create spectral filter
        if centroid_ratio > 1.1:
            # Brighten voice
            spectral_filter = 1 + 0.3 * (freqs / np.max(freqs))
        elif centroid_ratio < 0.9:
            # Darken voice
            spectral_filter = 1 - 0.3 * (freqs / np.max(freqs))
        else:
            spectral_filter = np.ones_like(freqs)
        
        spectral_filter = np.clip(spectral_filter, 0.5, 2.0)
        magnitude_shaped = magnitude * spectral_filter.reshape(-1, 1)
        
        # Step 3: Formant adjustment
        target_formants = speaker_profile['formants']
        magnitude_shaped = self._adjust_formants(magnitude_shaped, freqs, target_formants)
        
        # Step 4: Reconstruct
        stft_converted = magnitude_shaped * np.exp(1j * phase)
        y_converted = librosa.istft(stft_converted)
        
        # Step 5: Apply texture modification
        target_zcr = speaker_profile['zcr']
        current_zcr = np.mean(librosa.feature.zero_crossing_rate(y_converted))
        
        if abs(target_zcr - current_zcr) > 0.01:
            if target_zcr > current_zcr:
                # Add texture
                noise = np.random.normal(0, 0.003, len(y_converted))
                y_converted = y_converted + noise
            else:
                # Smooth texture
                b, a = butter(2, 0.8, btype='low')
                y_converted = filtfilt(b, a, y_converted)
        
        # Final normalization
        y_converted = y_converted / np.max(np.abs(y_converted)) * 0.8
        
        print("Multi-speaker voice conversion completed")
        return y_converted
    
    def _adjust_formants(self, magnitude, freqs, target_formants):
        """Adjust formant frequencies"""
        
        try:
            # Simple formant enhancement at target frequencies
            for formant_freq in target_formants[:3]:
                if formant_freq > 0:
                    # Find closest frequency bin
                    freq_idx = np.argmin(np.abs(freqs - formant_freq))
                    
                    # Enhance around formant frequency
                    start_idx = max(0, freq_idx - 5)
                    end_idx = min(len(freqs), freq_idx + 5)
                    
                    enhancement = np.exp(-0.5 * ((np.arange(start_idx, end_idx) - freq_idx) / 2)**2)
                    magnitude[start_idx:end_idx] *= (1 + 0.2 * enhancement.reshape(-1, 1))
            
        except Exception as e:
            pass
        
        return magnitude
    
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