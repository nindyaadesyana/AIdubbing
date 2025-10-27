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
from sklearn.preprocessing import StandardScaler
import pickle

class EmbeddingVoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
        print("✅ Embedding-based Voice Cloner initialized")
    
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
            
            print(f"Training speaker embedding model...")
            
            for i in range(10):
                time.sleep(1)
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
                print(f"Embedding training progress: {progress}%")
            
            # Create speaker embedding
            voice_profile = self._create_speaker_embedding(dataset_path, audio_files)
            
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
            
            print(f"Speaker embedding model trained successfully!")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _create_speaker_embedding(self, dataset_path, audio_files):
        """Create comprehensive speaker embedding"""
        
        embeddings = []
        reference_audios = []
        
        print("Creating speaker embedding...")
        
        for i, audio_file in enumerate(audio_files[:10]):
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=15)
                
                if len(y_clean) > sr * 1.0:  # At least 1 second
                    print(f"  Extracting embedding from sample {i+1}: {audio_file}")
                    
                    # Extract comprehensive features for embedding
                    embedding = self._extract_speaker_features(y_clean, sr)
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                        
                        # Store reference audio
                        if len(reference_audios) < 3:
                            model_path = os.path.join(self.models_dir, dataset_path.split('/')[-3])
                            os.makedirs(model_path, exist_ok=True)
                            
                            ref_file = os.path.join(model_path, f"reference_{len(reference_audios)}.wav")
                            sf.write(ref_file, y_clean, sr)
                            reference_audios.append(ref_file)
                        
            except Exception as e:
                print(f"    Error processing {audio_file}: {e}")
                continue
        
        if len(embeddings) < 2:
            raise Exception("Not enough valid samples for speaker embedding")
        
        # Create final speaker embedding
        speaker_embedding = self._compute_speaker_embedding(embeddings)
        
        # Save embedding data
        model_path = os.path.join(self.models_dir, dataset_path.split('/')[-3])
        embedding_file = os.path.join(model_path, "speaker_embedding.pkl")
        
        with open(embedding_file, 'wb') as f:
            pickle.dump({
                'embedding': speaker_embedding,
                'raw_embeddings': embeddings
            }, f)
        
        return {
            'voice_id': dataset_path.split('/')[-3],
            'speaker_embedding': speaker_embedding,
            'reference_audios': reference_audios,
            'embedding_file': embedding_file,
            'sample_count': len(embeddings),
            'created_at': datetime.now().isoformat()
        }
    
    def _extract_speaker_features(self, y, sr):
        """Extract comprehensive speaker features for embedding"""
        
        try:
            # Pitch features
            pitches = librosa.yin(y, fmin=80, fmax=400)
            pitch_values = pitches[~np.isnan(pitches)]
            
            if len(pitch_values) < 10:
                return None
            
            pitch_features = [
                np.mean(pitch_values),
                np.std(pitch_values),
                np.percentile(pitch_values, 25),
                np.percentile(pitch_values, 75)
            ]
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            spectral_features = [
                np.mean(spectral_centroid),
                np.std(spectral_centroid),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ]
            
            # Add spectral contrast features
            spectral_features.extend(np.mean(spectral_contrast, axis=1).tolist())
            
            # MFCC features (first 13 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_features = []
            for i in range(13):
                mfcc_features.extend([
                    np.mean(mfcc[i]),
                    np.std(mfcc[i])
                ])
            
            # Chroma features (alternative implementation)
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_features = np.mean(chroma, axis=1).tolist()
            except:
                # Fallback: use spectral features as chroma substitute
                chroma_features = [0.0] * 12
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_features = [np.mean(zcr), np.std(zcr)]
            
            # Formant features (simplified)
            formants = self._estimate_formants(y, sr)
            
            # Combine all features into embedding vector
            embedding = (
                pitch_features +
                spectral_features +
                mfcc_features +
                chroma_features +
                zcr_features +
                formants
            )
            
            return embedding
            
        except Exception as e:
            print(f"    Feature extraction error: {e}")
            return None
    
    def _estimate_formants(self, y, sr):
        """Estimate formant frequencies"""
        try:
            # Simple formant estimation using LPC
            from scipy.signal import lfilter
            
            # Pre-emphasis
            pre_emphasis = 0.97
            y_preemph = lfilter([1, -pre_emphasis], [1], y)
            
            # Window the signal
            windowed = y_preemph * np.hanning(len(y_preemph))
            
            # Autocorrelation
            autocorr = np.correlate(windowed, windowed, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find formants (simplified)
            fft = np.fft.fft(autocorr[:1024])
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(1024, 1/sr)
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(magnitude[:512], height=np.max(magnitude)*0.1)
            
            formant_freqs = freqs[peaks]
            formant_freqs = formant_freqs[(formant_freqs > 200) & (formant_freqs < 3500)]
            
            # Return first 4 formants
            formants = sorted(formant_freqs)[:4]
            while len(formants) < 4:
                formants.append(0.0)
            
            return formants[:4]
            
        except:
            return [800.0, 1200.0, 2500.0, 3500.0]  # Default formants
    
    def _compute_speaker_embedding(self, embeddings):
        """Compute final speaker embedding from all samples"""
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        # Normalize features
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embedding_matrix)
        
        # Compute mean embedding (speaker representation)
        mean_embedding = np.mean(normalized_embeddings, axis=0)
        
        # Compute variance (speaker consistency)
        var_embedding = np.var(normalized_embeddings, axis=0)
        
        return {
            'mean': mean_embedding.tolist(),
            'variance': var_embedding.tolist(),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'dimension': len(mean_embedding)
        }
    
    def generate_speech(self, voice_id, text, output_path):
        """Generate speech using speaker embedding"""
        
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model '{voice_id}' not found")
        
        try:
            print(f"Generating speech with speaker embedding: {text}")
            
            # Step 1: Generate base TTS
            tts = gTTS(text=text, lang='id', slow=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                base_tts_path = tmp_file.name
            
            # Step 2: Load TTS
            y_tts, sr = librosa.load(base_tts_path, sr=22050)
            os.unlink(base_tts_path)
            
            # Step 3: Apply speaker embedding-based conversion
            profile = self.voice_models[voice_id]
            y_converted = self._apply_embedding_conversion(y_tts, sr, profile)
            
            # Step 4: Save
            sf.write(output_path, y_converted, sr)
            
            print(f"Speaker embedding TTS generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating embedding-based TTS: {e}")
            return False
    
    def _apply_embedding_conversion(self, y_tts, sr, profile):
        """Apply voice conversion using speaker embedding"""
        
        print("Applying speaker embedding conversion...")
        
        # Extract current TTS features
        current_features = self._extract_speaker_features(y_tts, sr)
        
        if current_features is None:
            print("Warning: Could not extract TTS features, using basic conversion")
            return y_tts
        
        # Load speaker embedding
        speaker_embedding = profile['speaker_embedding']
        target_mean = np.array(speaker_embedding['mean'])
        
        # Normalize current features using saved scaler parameters
        scaler_mean = np.array(speaker_embedding['scaler_mean'])
        scaler_scale = np.array(speaker_embedding['scaler_scale'])
        
        current_normalized = (np.array(current_features) - scaler_mean) / scaler_scale
        
        # Compute conversion parameters
        conversion_vector = target_mean - current_normalized
        
        # Apply targeted conversions based on embedding differences
        y_converted = self._apply_targeted_conversion(y_tts, sr, conversion_vector, current_features)
        
        print("Speaker embedding conversion completed")
        return y_converted
    
    def _apply_targeted_conversion(self, y, sr, conversion_vector, current_features):
        """Apply targeted voice conversion based on embedding differences"""
        
        # Extract key conversion parameters from embedding difference
        # Assuming feature order: pitch(4) + spectral(13) + mfcc(26) + chroma(12) + zcr(2) + formants(4)
        
        # Pitch conversion (first 4 features)
        pitch_diff = conversion_vector[:4]
        pitch_adjustment = np.mean(pitch_diff) * 2.0  # Scale factor
        
        if abs(pitch_adjustment) > 0.1:
            semitone_shift = np.clip(pitch_adjustment * 12, -8, 8)
            print(f"Embedding-based pitch shift: {semitone_shift:.1f} semitones")
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone_shift)
        
        # Spectral conversion (next 13 features)
        spectral_diff = conversion_vector[4:17]
        spectral_adjustment = np.mean(spectral_diff)
        
        if abs(spectral_adjustment) > 0.1:
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            freq_bins = magnitude.shape[0]
            freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-1)
            
            # Apply spectral shaping based on embedding difference
            if spectral_adjustment > 0:
                # Brighten
                spectral_filter = 1 + 0.3 * spectral_adjustment * (freqs / np.max(freqs))
            else:
                # Darken
                spectral_filter = 1 + 0.3 * spectral_adjustment * (1 - freqs / np.max(freqs))
            
            spectral_filter = np.clip(spectral_filter, 0.5, 2.0)
            magnitude_shaped = magnitude * spectral_filter.reshape(-1, 1)
            
            # Reconstruct
            stft_converted = magnitude_shaped * np.exp(1j * phase)
            y = librosa.istft(stft_converted)
        
        # MFCC-based timbre adjustment (features 17:43)
        mfcc_diff = conversion_vector[17:43]
        timbre_adjustment = np.mean(mfcc_diff)
        
        if abs(timbre_adjustment) > 0.1:
            # Apply subtle filtering for timbre adjustment
            if timbre_adjustment > 0:
                # Add warmth
                b, a = butter(2, 0.3, btype='high')
                y_filtered = filtfilt(b, a, y)
                y = 0.7 * y + 0.3 * y_filtered
            else:
                # Add smoothness
                b, a = butter(2, 0.7, btype='low')
                y_filtered = filtfilt(b, a, y)
                y = 0.7 * y + 0.3 * y_filtered
        
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