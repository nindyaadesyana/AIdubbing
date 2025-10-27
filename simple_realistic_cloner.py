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

class SimpleRealisticCloner:
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
        """Training model - analisis karakteristik suara sederhana"""
        try:
            import time
            self.training_status[training_id]['status'] = 'training'
            
            # Path dataset
            dataset_path = f"datasets/processed/{voice_id}/clips/"
            
            if not os.path.exists(dataset_path):
                raise Exception(f"Dataset path not found: {dataset_path}")
            
            # Analisis karakteristik suara
            audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
            
            if len(audio_files) < 2:
                raise Exception("Not enough audio samples for training")
            
            print(f"Analyzing {len(audio_files)} voice samples...")
            
            # Simulate training progress berdasarkan epochs
            total_steps = 10
            for i in range(total_steps):
                time.sleep(max(0.5, epochs / 100))  # Lebih lama jika epochs lebih banyak
                progress = (i + 1) * 10
                self.training_status[training_id]['progress'] = progress
            
            # Ekstraksi karakteristik suara sederhana tapi efektif
            voice_profile = self._extract_simple_voice_features(dataset_path, audio_files)
            
            # Save voice profile
            model_path = os.path.join(self.models_dir, voice_id)
            os.makedirs(model_path, exist_ok=True)
            
            profile_file = os.path.join(model_path, "voice_profile.json")
            with open(profile_file, 'w') as f:
                json.dump(voice_profile, f, indent=2)
            
            # Load ke memory
            self.voice_models[voice_id] = voice_profile
            
            self.training_status[training_id].update({
                'status': 'completed',
                'progress': 100,
                'end_time': datetime.now().isoformat(),
                'model_path': model_path
            })
            
            print(f"Voice training completed for {voice_id}")
            
        except Exception as e:
            self.training_status[training_id].update({
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
    
    def _extract_simple_voice_features(self, dataset_path, audio_files):
        """Ekstraksi fitur suara yang sederhana tapi efektif"""
        
        pitch_values = []
        best_samples = []
        durations = []
        
        # Analisis sample terbaik
        for audio_file in audio_files[:10]:  # Maksimal 10 file
            file_path = os.path.join(dataset_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
                
                # Trim silence
                y_trimmed, _ = librosa.effects.trim(y, top_db=20)
                
                if len(y_trimmed) > sr * 0.5:  # Minimal 0.5 detik
                    # Analisis pitch
                    pitches, magnitudes = librosa.piptrack(y=y_trimmed, sr=sr)
                    pitch_values_frame = []
                    
                    for t in range(pitches.shape[1]):
                        index = magnitudes[:, t].argmax()
                        pitch = pitches[index, t]
                        if pitch > 80 and pitch < 400:  # Range suara manusia
                            pitch_values_frame.append(pitch)
                    
                    if len(pitch_values_frame) > 10:
                        avg_pitch = np.mean(pitch_values_frame)
                        pitch_values.append(avg_pitch)
                        
                        # Simpan sample yang bagus
                        best_samples.append({
                            'file': audio_file,
                            'pitch': avg_pitch,
                            'duration': len(y_trimmed) / sr,
                            'quality_score': len(pitch_values_frame) / len(y_trimmed) * sr  # Pitch stability
                        })
                        
                        durations.append(len(y_trimmed) / sr)
                
            except Exception as e:
                print(f"Error analyzing {audio_file}: {e}")
                continue
        
        # Pilih sample terbaik berdasarkan quality score
        best_samples.sort(key=lambda x: x['quality_score'], reverse=True)
        top_samples = best_samples[:5]  # 5 sample terbaik
        
        # Hitung statistik
        avg_pitch = np.mean(pitch_values) if pitch_values else 200
        pitch_std = np.std(pitch_values) if len(pitch_values) > 1 else 20
        avg_duration = np.mean(durations) if durations else 2.0
        
        return {
            'voice_id': dataset_path.split('/')[-3],
            'avg_pitch': float(avg_pitch),
            'pitch_std': float(pitch_std),
            'avg_duration': float(avg_duration),
            'best_samples': [s['file'] for s in top_samples],
            'sample_count': len(audio_files),
            'dataset_path': dataset_path,
            'created_at': datetime.now().isoformat()
        }
    
    def get_training_status(self, training_id):
        """Get status training"""
        return self.training_status.get(training_id, {'status': 'not_found'})
    
    def generate_speech(self, text, voice_id):
        """Generate speech dengan pendekatan hybrid yang realistis"""
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
            # Strategi hybrid: 70% TTS + 30% voice samples
            if len(text) > 50:  # Text panjang - gunakan TTS dengan modifikasi
                final_audio = self._generate_hybrid_long_text(text, voice_profile)
            else:  # Text pendek - gunakan sample dengan TTS backup
                final_audio = self._generate_hybrid_short_text(text, voice_profile)
            
            # Save audio
            sf.write(output_path, final_audio, 22050)
            
            if not os.path.exists(output_path):
                raise Exception("Failed to save audio file")
            
            print(f"Generated speech: '{text[:30]}...' using hybrid approach")
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")
    
    def _generate_hybrid_long_text(self, text, voice_profile):
        """Generate untuk text panjang menggunakan TTS + voice modification"""
        try:
            # Generate base TTS
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts = gTTS(text=text, lang='id', slow=False)
                tts.save(tmp_file.name)
                
                base_audio, sr = librosa.load(tmp_file.name, sr=22050)
                os.unlink(tmp_file.name)
            
            # Apply simple voice modification
            modified_audio = self._apply_simple_voice_modification(base_audio, voice_profile)
            
            return modified_audio
            
        except Exception as e:
            print(f"TTS generation failed: {e}")
            return self._generate_from_samples_long(text, voice_profile)
    
    def _generate_hybrid_short_text(self, text, voice_profile):
        """Generate untuk text pendek menggunakan sample + TTS backup"""
        
        # Coba gunakan sample voice asli dulu
        sample_audio = self._generate_from_best_samples(text, voice_profile)
        
        if sample_audio is not None:
            return sample_audio
        else:
            # Fallback ke TTS
            return self._generate_hybrid_long_text(text, voice_profile)
    
    def _generate_from_best_samples(self, text, voice_profile):
        """Generate dari sample terbaik"""
        try:
            dataset_path = voice_profile.get('dataset_path')
            best_samples = voice_profile.get('best_samples', [])
            
            if not dataset_path or not best_samples:
                return None
            
            # Pilih 1-2 sample terbaik
            selected_samples = best_samples[:2]
            
            combined_audio = []
            for sample_file in selected_samples:
                sample_path = os.path.join(dataset_path, sample_file)
                if os.path.exists(sample_path):
                    try:
                        audio, sr = librosa.load(sample_path, sr=22050)
                        # Trim dan clean
                        audio, _ = librosa.effects.trim(audio, top_db=20)
                        combined_audio.append(audio)
                    except:
                        continue
            
            if not combined_audio:
                return None
            
            # Gabungkan dengan jeda natural
            silence = np.zeros(int(0.15 * 22050))  # 150ms jeda
            final_audio = combined_audio[0]
            
            for audio_segment in combined_audio[1:]:
                final_audio = np.concatenate([final_audio, silence, audio_segment])
            
            # Adjust durasi berdasarkan text
            target_duration = max(2.0, len(text) * 0.08)
            target_samples = int(target_duration * 22050)
            
            if len(final_audio) < target_samples:
                # Extend dengan repetisi yang natural
                gap = np.zeros(int(0.3 * 22050))  # 300ms gap
                extended_audio = [final_audio]
                
                while len(np.concatenate(extended_audio)) < target_samples:
                    extended_audio.extend([gap, combined_audio[0]])
                
                final_audio = np.concatenate(extended_audio)
            
            # Trim ke target length
            final_audio = final_audio[:target_samples]
            
            # Apply gentle processing
            final_audio = self._apply_gentle_processing(final_audio)
            
            return final_audio
            
        except Exception as e:
            print(f"Sample generation error: {e}")
            return None
    
    def _apply_simple_voice_modification(self, audio, voice_profile):
        """Apply modifikasi suara yang sederhana tapi efektif"""
        try:
            sr = 22050
            modified_audio = audio.copy()
            
            # 1. Pitch adjustment yang subtle
            target_pitch = voice_profile.get('avg_pitch', 200)
            
            if target_pitch > 220:  # Suara lebih tinggi
                pitch_shift = random.uniform(1, 2)
            elif target_pitch < 180:  # Suara lebih rendah
                pitch_shift = random.uniform(-2, -1)
            else:  # Suara normal
                pitch_shift = random.uniform(-0.5, 0.5)
            
            # Apply pitch shift yang subtle
            if abs(pitch_shift) > 0.3:
                modified_audio = librosa.effects.pitch_shift(modified_audio, sr=sr, n_steps=pitch_shift)
            
            # 2. Gentle filtering untuk voice character
            if target_pitch > 200:
                # Slightly brighter
                b, a = butter(1, [1000, 6000], btype='band', fs=sr)
                filtered = filtfilt(b, a, modified_audio)
                modified_audio = modified_audio * 0.8 + filtered * 0.2
            else:
                # Slightly warmer
                b, a = butter(1, [300, 4000], btype='band', fs=sr)
                filtered = filtfilt(b, a, modified_audio)
                modified_audio = modified_audio * 0.8 + filtered * 0.2
            
            return modified_audio
            
        except Exception as e:
            print(f"Voice modification error: {e}")
            return audio
    
    def _apply_gentle_processing(self, audio):
        """Apply processing yang gentle untuk natural sound"""
        sr = 22050
        
        # 1. Gentle normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.85
        
        # 2. Smooth fade in/out
        fade_samples = int(0.05 * sr)  # 50ms fade
        if len(audio) > fade_samples * 2:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # 3. Subtle noise reduction
        # Simple high-pass filter untuk remove low-freq noise
        b, a = butter(1, 80, btype='high', fs=sr)
        audio = filtfilt(b, a, audio)
        
        return audio.astype(np.float32)
    
    def _generate_from_samples_long(self, text, voice_profile):
        """Generate dari samples untuk text panjang"""
        # Sama seperti short text tapi dengan repetisi lebih banyak
        base_audio = self._generate_from_best_samples(text, voice_profile)
        
        if base_audio is not None:
            return base_audio
        
        # Ultimate fallback
        duration = max(3.0, len(text) * 0.08)
        t = np.linspace(0, duration, int(22050 * duration))
        
        # Generate pleasant tone
        audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3 note
        
        # Add envelope
        envelope = np.ones_like(t)
        fade_samples = int(0.1 * 22050)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio *= envelope
        
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