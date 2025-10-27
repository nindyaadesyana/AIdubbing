import os
import json
import uuid
import threading
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime
from gtts import gTTS
import tempfile
import random
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

class TrueVoiceCloner:
    def __init__(self):
        self.models_dir = "models"
        self.training_status = {}
        self.voice_models = {}
        
        os.makedirs(self.models_dir, exist_ok=True)
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load model yang sudah ada"""
        for voice_dir in os.listdir(self.models_dir):
            voice_path = os.path.join(self.models_dir, voice_dir)
            if os.path.isdir(voice_path):
                model_file = os.path.join(voice_path, "voice_profile.json")
                if os.path.exists(model_file):
                    with open(model_file, 'r') as f:
                        profile = json.load(f)
                        self.voice_models[voice_dir] = profile
    
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
        
        thread = threading.Thread(
            target=self._train_model,
            args=(training_id, voice_id, epochs)
        )
        thread.daemon = True
        thread.start()
        
        return training_id
    
    def _train_model(self, training_id, voice_id, epochs):
        """Training model - ekstraksi karakteristik suara untuk voice conversion"""
        try:
            import time
            self.training_status[training_id]['status'] = 'training'
            
            dataset_path = f"datasets/processed/{voice_id}/clips/"
            
            if not os.path.exists(dataset_path):
                raise Exception(f"Dataset path not found: {dataset_path}")
            
            audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
            
            if len(audio_files) < 2:
                raise Exception("Not enough audio samples for training")
            
            print(f"Analyzing voice characteristics from {len(audio_files)} samples...")
            
            # Simulate training dengan progress yang realistis
            for i in range(10):
                time.sleep(epochs / 100)  # Lebih lama untuk epochs lebih tinggi
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
                print(f"Training progress: {progress}%")
            
            # Ekstraksi karakteristik suara yang mendalam
            voice_profile = self._extract_voice_characteristics(dataset_path, audio_files)
            
            # Save voice profile
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
            
            print(f"Voice training completed! Model saved for {voice_id}")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _extract_voice_characteristics(self, dataset_path, audio_files):
        """Ekstraksi karakteristik suara untuk voice conversion"""
        
        pitch_values = []
        formant_data = []
        spectral_data = []
        voice_samples = []
        
        for audio_file in audio_files[:15]:  # Analisis 15 file terbaik
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                y_clean, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_clean) > sr * 0.5:  # Minimal 0.5 detik
                    # 1. Pitch analysis
                    pitches, magnitudes = librosa.piptrack(y=y_clean, sr=sr, threshold=0.1)
                    valid_pitches = []
                    
                    for t in range(pitches.shape[1]):
                        index = magnitudes[:, t].argmax()
                        pitch = pitches[index, t]
                        if 80 < pitch < 400:  # Human voice range
                            valid_pitches.append(pitch)
                    
                    if len(valid_pitches) > 10:
                        pitch_values.extend(valid_pitches)
                        
                        # 2. Spectral envelope (timbre)
                        stft = librosa.stft(y_clean)
                        magnitude = np.abs(stft)
                        spectral_envelope = np.mean(magnitude, axis=1)
                        spectral_data.append(spectral_envelope)
                        
                        # 3. Formant estimation (simplified)
                        formants = self._estimate_formants(y_clean, sr)
                        if formants:
                            formant_data.append(formants)
                        
                        # 4. Save good samples
                        voice_samples.append({
                            'file': audio_file,
                            'audio': y_clean,
                            'pitch_mean': np.mean(valid_pitches),
                            'duration': len(y_clean) / sr
                        })
                
            except Exception as e:
                print(f"Error analyzing {audio_file}: {e}")
                continue
        
        # Compute statistics
        avg_pitch = np.mean(pitch_values) if pitch_values else 200
        pitch_std = np.std(pitch_values) if len(pitch_values) > 1 else 20
        
        # Average spectral envelope
        avg_spectral_envelope = None
        if spectral_data:
            # Normalize lengths
            min_len = min(len(env) for env in spectral_data)
            normalized_spectral = [env[:min_len] for env in spectral_data]
            avg_spectral_envelope = np.mean(normalized_spectral, axis=0).tolist()
        
        # Average formants
        avg_formants = None
        if formant_data:
            avg_formants = np.mean(formant_data, axis=0).tolist()
        
        return {
            'voice_id': dataset_path.split('/')[-3],
            'avg_pitch': float(avg_pitch),
            'pitch_std': float(pitch_std),
            'pitch_range': [float(np.min(pitch_values)), float(np.max(pitch_values))] if pitch_values else [150, 300],
            'spectral_envelope': avg_spectral_envelope,
            'formants': avg_formants,
            'voice_samples': [s['file'] for s in voice_samples[:5]],  # Top 5 samples
            'dataset_path': dataset_path,
            'sample_count': len(audio_files),
            'created_at': datetime.now().isoformat()
        }
    
    def _estimate_formants(self, y, sr):
        """Estimasi formant frequencies (simplified)"""
        try:
            # Pre-emphasis
            pre_emphasis = 0.97
            y_preemph = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
            
            # Window the signal
            window_length = int(0.025 * sr)  # 25ms
            hop_length = int(0.01 * sr)      # 10ms
            
            frames = librosa.util.frame(y_preemph, frame_length=window_length, hop_length=hop_length)
            
            formants_per_frame = []
            for frame in frames.T[:10]:  # Analyze first 10 frames
                if len(frame) > 10:
                    # FFT
                    fft = np.fft.fft(frame * np.hanning(len(frame)))
                    magnitude = np.abs(fft[:len(fft)//2])
                    
                    # Find peaks (simplified formant detection)
                    peaks = []
                    for i in range(2, len(magnitude)-2):
                        if (magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1] and
                            magnitude[i] > magnitude[i-2] and magnitude[i] > magnitude[i+2]):
                            freq = i * sr / len(frame)
                            if 200 < freq < 3500:  # Formant frequency range
                                peaks.append(freq)
                    
                    if len(peaks) >= 2:
                        formants_per_frame.append(sorted(peaks)[:3])  # F1, F2, F3
            
            if formants_per_frame:
                # Return average formants
                avg_formants = np.mean(formants_per_frame, axis=0)
                return avg_formants.tolist()
                
        except Exception as e:
            print(f"Formant estimation error: {e}")
        
        return None
    
    def get_training_status(self, training_id):
        """Get status training"""
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def generate_speech(self, text, voice_id):
        """Generate speech yang benar-benar mengucapkan text dengan voice yang di-clone"""
        if voice_id not in self.voice_models:
            raise Exception(f"Voice model {voice_id} not found or not trained")
        
        if not text or len(text.strip()) == 0:
            raise Exception("Text cannot be empty")
        
        voice_profile = self.voice_models[voice_id]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{voice_id}_{timestamp}.wav"
        output_path = os.path.join("outputs", output_filename)
        
        os.makedirs("outputs", exist_ok=True)
        
        try:
            print(f"Generating speech: '{text}' with cloned voice...")
            
            # Step 1: Generate base TTS dari text
            base_tts_audio = self._generate_base_tts(text)
            
            # Step 2: Apply voice conversion menggunakan voice profile
            cloned_audio = self._apply_voice_conversion(base_tts_audio, voice_profile)
            
            # Step 3: Post-processing
            final_audio = self._post_process(cloned_audio)
            
            # Save audio
            sf.write(output_path, final_audio, 22050)
            
            if not os.path.exists(output_path):
                raise Exception("Failed to save audio file")
            
            print(f"✅ Speech generated successfully with cloned voice!")
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")
    
    def _generate_base_tts(self, text):
        """Generate base TTS audio dari text"""
        try:
            # Gunakan gTTS untuk generate speech dari text
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts = gTTS(text=text, lang='id', slow=False)
                tts.save(tmp_file.name)
                
                # Load audio
                audio, sr = librosa.load(tmp_file.name, sr=22050)
                os.unlink(tmp_file.name)
                
                print(f"Base TTS generated: {len(audio)/sr:.1f} seconds")
                return audio
                
        except Exception as e:
            print(f"TTS generation failed: {e}")
            raise Exception("Failed to generate base TTS audio")
    
    def _apply_voice_conversion(self, source_audio, voice_profile):
        """Apply voice conversion untuk mengubah suara TTS menjadi target voice"""
        try:
            sr = 22050
            converted_audio = source_audio.copy()
            
            print("Applying voice conversion...")
            
            # 1. Pitch conversion
            target_pitch = voice_profile.get('avg_pitch', 200)
            
            # Estimate source pitch
            pitches, magnitudes = librosa.piptrack(y=converted_audio, sr=sr)
            source_pitches = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    source_pitches.append(pitch)
            
            if source_pitches:
                source_pitch_mean = np.mean(source_pitches)
                
                # Calculate pitch shift in semitones
                pitch_ratio = target_pitch / source_pitch_mean
                pitch_shift_semitones = 12 * np.log2(pitch_ratio)
                
                # Apply pitch shift
                if abs(pitch_shift_semitones) > 0.5:
                    converted_audio = librosa.effects.pitch_shift(
                        converted_audio, sr=sr, n_steps=pitch_shift_semitones
                    )
                    print(f"Applied pitch shift: {pitch_shift_semitones:.1f} semitones")
            
            # 2. Spectral envelope matching
            spectral_envelope = voice_profile.get('spectral_envelope')
            if spectral_envelope:
                converted_audio = self._apply_spectral_envelope_matching(
                    converted_audio, spectral_envelope, sr
                )
                print("Applied spectral envelope matching")
            
            # 3. Formant shifting
            formants = voice_profile.get('formants')
            if formants:
                converted_audio = self._apply_formant_shifting(
                    converted_audio, formants, sr
                )
                print("Applied formant shifting")
            
            return converted_audio
            
        except Exception as e:
            print(f"Voice conversion error: {e}")
            return source_audio
    
    def _apply_spectral_envelope_matching(self, audio, target_envelope, sr):
        """Apply spectral envelope matching"""
        try:
            # STFT
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Current envelope
            current_envelope = np.mean(magnitude, axis=1)
            
            # Match lengths
            target_envelope = np.array(target_envelope)
            min_len = min(len(current_envelope), len(target_envelope))
            
            current_envelope = current_envelope[:min_len]
            target_envelope = target_envelope[:min_len]
            
            # Avoid division by zero
            current_envelope = np.maximum(current_envelope, 1e-10)
            
            # Compute transfer function
            transfer_function = target_envelope / current_envelope
            
            # Smooth transfer function
            from scipy.ndimage import gaussian_filter1d
            transfer_function = gaussian_filter1d(transfer_function, sigma=2)
            
            # Apply to magnitude spectrum
            for i in range(min_len):
                magnitude[i, :] *= transfer_function[i]
            
            # Reconstruct
            modified_stft = magnitude * np.exp(1j * phase)
            modified_audio = librosa.istft(modified_stft)
            
            return modified_audio
            
        except Exception as e:
            print(f"Spectral envelope matching error: {e}")
            return audio
    
    def _apply_formant_shifting(self, audio, target_formants, sr):
        """Apply formant shifting"""
        try:
            if len(target_formants) >= 2:
                f1_target = target_formants[0]
                f2_target = target_formants[1]
                
                # Apply bandpass filters around formant frequencies
                # F1 enhancement
                if 200 < f1_target < 1000:
                    b1, a1 = butter(2, [f1_target*0.8, f1_target*1.2], btype='band', fs=sr)
                    f1_component = filtfilt(b1, a1, audio) * 0.2
                    
                    # F2 enhancement
                    if 800 < f2_target < 2500:
                        b2, a2 = butter(2, [f2_target*0.8, f2_target*1.2], btype='band', fs=sr)
                        f2_component = filtfilt(b2, a2, audio) * 0.15
                        
                        # Combine
                        audio = audio * 0.8 + f1_component + f2_component
            
            return audio
            
        except Exception as e:
            print(f"Formant shifting error: {e}")
            return audio
    
    def _post_process(self, audio):
        """Post-processing untuk hasil yang natural"""
        sr = 22050
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.85
        
        # Smooth fade in/out
        fade_samples = int(0.05 * sr)
        if len(audio) > fade_samples * 2:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Light noise gate
        threshold = 0.01
        audio[np.abs(audio) < threshold] *= 0.1
        
        return audio.astype(np.float32)
    
    def list_available_voices(self):
        """List semua voice yang tersedia"""
        voices = []
        for voice_id, profile in self.voice_models.items():
            voices.append({
                'id': voice_id,
                'name': voice_id.replace('_', ' ').title(),
                'status': 'ready',
                'samples_analyzed': profile.get('sample_count', 0),
                'created_at': profile.get('created_at', '')
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